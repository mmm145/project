import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import ray
import torch
import vllm
from blingfire import text_to_sentences_and_offsets
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from tqdm import tqdm
import re
import html
from urllib.parse import urlparse

#### CONFIG PARAMETERS ---

# Define the number of context sentences to consider for generating an answer.
NUM_CONTEXT_SENTENCES = 30  # Increased from 20 in the baseline
# Set the maximum length for each context sentence (in characters).
MAX_CONTEXT_SENTENCE_LENGTH = 1000
# Set the maximum context references length (in characters).
MAX_CONTEXT_REFERENCES_LENGTH = 6000  # Increased from 4000 in the baseline

# Batch size you wish the evaluators will use to call the `batch_generate_answer` function
AICROWD_SUBMISSION_BATCH_SIZE = 1

# VLLM Parameters
VLLM_TENSOR_PARALLEL_SIZE = 1
VLLM_GPU_MEMORY_UTILIZATION = 0.85

# Sentence Transformer Parameters - use a more powerful model for better embedding
SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-mpnet-base-v2"  # Better than all-MiniLM-L6-v2
SENTENCE_TRANSFORMER_BATCH_SIZE = 32

#### CONFIG PARAMETERS END---

class ChunkExtractor:
    """Enhanced chunk extractor with more sophisticated processing."""

    @ray.remote
    def _extract_chunks(self, interaction_id, html_source, url=""):
        """
        Extracts and returns chunks from given HTML source with improved text cleaning.

        Parameters:
            interaction_id (str): Interaction ID that this HTML source belongs to.
            html_source (str): HTML content from which to extract text.
            url (str): URL of the source document (if available) for metadata extraction.

        Returns:
            Tuple[str, List[str], str]: A tuple containing the interaction ID, a list of sentences, 
            and metadata (like URL domain, date if available).
        """
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(html_source, "lxml")
        
        # Remove script and style elements that might interfere with text extraction
        for script_or_style in soup(["script", "style", "noscript", "iframe"]):
            script_or_style.decompose()
            
        # Check for main, article, or content divs that might contain the primary content
        main_content = None
        for content_element in ['main', 'article', 'div[role="main"]', 'div.content', 'div.main']:
            content = soup.select_one(content_element)
            if content:
                main_content = content
                break
                
        if main_content:
            # Use the identified main content
            text = main_content.get_text(" ", strip=True)
        else:
            # Fall back to the entire document
            text = soup.get_text(" ", strip=True)
        
        # Clean text: normalize whitespace, remove excessive newlines
        text = re.sub(r'\s+', ' ', text)
        text = html.unescape(text)  # Convert HTML entities to their actual characters
        
        if not text:
            # Return a list with empty string when no text is extracted
            return interaction_id, [""], ""
            
        # Extract metadata
        metadata = ""
        if url:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            metadata = f"Source: {domain}"
            
            # Look for date information in the page
            date_tags = soup.select("time") or soup.find_all(attrs={"itemprop": "datePublished"})
            if date_tags:
                published_date = date_tags[0].get("datetime", "") or date_tags[0].text.strip()
                if published_date:
                    metadata += f", Published: {published_date}"

        # Extract offsets of sentences from the text
        _, offsets = text_to_sentences_and_offsets(text)

        # Initialize a list to store sentences
        chunks = []

        # Iterate through the list of offsets and extract sentences
        for start, end in offsets:
            # Extract the sentence and limit its length
            sentence = text[start:end][:MAX_CONTEXT_SENTENCE_LENGTH]
            # Only add sentences that contain meaningful content
            if len(sentence.strip()) > 10:  # Skip very short fragments
                chunks.append(sentence)
                
        # Extract structured data like tables if present
        tables = soup.find_all("table")
        for table in tables[:3]:  # Limit to first few tables to avoid overwhelming context
            table_text = "TABLE: "
            rows = table.find_all("tr")
            for i, row in enumerate(rows[:10]):  # Limit rows
                cells = row.find_all(["td", "th"])
                row_content = " | ".join([cell.get_text(strip=True) for cell in cells])
                if row_content.strip():
                    table_text += f"{row_content}; "
            if len(table_text) > 20:  # Only add non-empty tables
                chunks.append(table_text[:MAX_CONTEXT_SENTENCE_LENGTH])

        return interaction_id, chunks, metadata

    def extract_chunks(self, batch_interaction_ids, batch_search_results):
        """
        Extracts chunks from given batch search results using parallel processing with Ray.

        Parameters:
            batch_interaction_ids (List[str]): List of interaction IDs.
            batch_search_results (List[List[Dict]]): List of search results batches, each containing HTML text.

        Returns:
            Tuple[np.ndarray, np.ndarray, Dict[str, str]]: A tuple containing an array of chunks, 
            an array of corresponding interaction IDs, and metadata dictionary.
        """
        # Setup parallel chunk extraction using ray remote
        ray_response_refs = []
        
        for idx, search_results in enumerate(batch_search_results):
            for html_result in search_results:
                url = html_result.get("url", "")
                ray_response_refs.append(
                    self._extract_chunks.remote(
                        self,
                        interaction_id=batch_interaction_ids[idx],
                        html_source=html_result["page_result"],
                        url=url
                    )
                )

        # Wait until all extractions are complete and collect chunks with metadata
        chunk_dictionary = defaultdict(list)
        metadata_dictionary = defaultdict(str)

        for response_ref in ray_response_refs:
            interaction_id, _chunks, metadata = ray.get(response_ref)
            chunk_dictionary[interaction_id].extend(_chunks)
            # Append metadata if not already present
            if metadata and metadata not in metadata_dictionary[interaction_id]:
                metadata_dictionary[interaction_id] += f"{metadata}; "

        # Flatten chunks and keep a map of corresponding interaction_ids
        chunks, chunk_interaction_ids = self._flatten_chunks(chunk_dictionary)

        return chunks, chunk_interaction_ids, metadata_dictionary

    def _flatten_chunks(self, chunk_dictionary):
        """
        Flattens the chunk dictionary with improved deduplication.

        Parameters:
            chunk_dictionary (defaultdict): Dictionary with interaction IDs as keys and lists of chunks as values.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing an array of chunks and an array of corresponding interaction IDs.
        """
        chunks = []
        chunk_interaction_ids = []

        for interaction_id, _chunks in chunk_dictionary.items():
            # More sophisticated deduplication that preserves sentence order
            # but removes duplicate or highly similar content
            unique_chunks = []
            seen_content = set()
            
            for chunk in _chunks:
                # Create a normalized version for comparison
                normalized = re.sub(r'\s+', ' ', chunk.lower().strip())
                # Skip if we've seen this exact content
                if normalized in seen_content:
                    continue
                    
                # Also check for high similarity with existing chunks
                is_similar = False
                for seen in seen_content:
                    # If one is a substring of the other with high overlap
                    if len(seen) > 20 and (seen in normalized or normalized in seen):
                        overlap_ratio = min(len(seen), len(normalized)) / max(len(seen), len(normalized))
                        if overlap_ratio > 0.8:  # High similarity threshold
                            is_similar = True
                            break
                
                if not is_similar:
                    seen_content.add(normalized)
                    unique_chunks.append(chunk)
            
            chunks.extend(unique_chunks)
            chunk_interaction_ids.extend([interaction_id] * len(unique_chunks))

        # Convert to numpy arrays for convenient slicing/masking operations later
        chunks = np.array(chunks)
        chunk_interaction_ids = np.array(chunk_interaction_ids)

        return chunks, chunk_interaction_ids


class EnhancedRAGModel:
    """
    Enhanced RAG Model that improves upon the baseline with:
    1. Better text extraction and chunk processing
    2. Improved embedding model
    3. Advanced query understanding and reformulation
    4. Source-aware context retrieval
    5. Sophisticated prompt engineering
    """
    def __init__(self, llm_name="meta-llama/Llama-3.2-3B-Instruct", is_server=False, vllm_server=None):
        self.initialize_models(llm_name, is_server, vllm_server)
        self.chunk_extractor = ChunkExtractor()
        self.query_cache = {}  # Cache for storing processed queries

    def initialize_models(self, llm_name, is_server, vllm_server):
        self.llm_name = llm_name
        self.is_server = is_server
        self.vllm_server = vllm_server

        if self.is_server:
            # initialize the model with vllm server
            openai_api_key = "EMPTY"
            openai_api_base = self.vllm_server
            self.llm_client = OpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base,
            )
        else:
            # initialize the model with vllm offline inference
            self.llm = vllm.LLM(
                model=self.llm_name,
                worker_use_ray=True,
                tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE,
                gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
                trust_remote_code=True,
                max_model_len=105216,
                dtype="half",
                enforce_eager=True
            )
            self.tokenizer = self.llm.get_tokenizer()

        # Load a better sentence transformer model for embeddings
        self.sentence_model = SentenceTransformer(
            SENTENCE_TRANSFORMER_MODEL,
            device=torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            ),
        )

    def calculate_embeddings(self, sentences):
        """
        Compute normalized embeddings with the enhanced model.
        """
        embeddings = self.sentence_model.encode(
            sentences=sentences,
            normalize_embeddings=True,
            batch_size=SENTENCE_TRANSFORMER_BATCH_SIZE,
        )
        return embeddings

    def parse_query(self, query: str, query_time: str) -> Dict:
        """
        Parse the query to extract important information and identify query type.
        
        Args:
            query: The user's query
            query_time: Timestamp when query was made
            
        Returns:
            Dict containing query information
        """
        # Cache key is the combination of query and time
        cache_key = f"{query}_{query_time}"
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]
            
        query_info = {
            'original_query': query,
            'query_time': query_time,
            'is_temporal': False,
            'has_condition': False,
            'is_comparative': False,
            'is_multi_hop': False,
            'question_type': 'factual',  # Default type
            'keywords': [],
            'entities': []
        }
        
        # Check for temporal indicators
        temporal_patterns = [
            r'(in|on|at|during|before|after|since) (january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)(\s+\d{1,2})?(\s*,\s*\d{4})?',
            r'(today|yesterday|last|current|previous|next|recent|upcoming|future|past)\s+(week|month|year|decade)',
            r'\b\d{4}\b',  # Years
            r'(now|currently|presently|at the moment)',
        ]
        
        for pattern in temporal_patterns:
            if re.search(pattern, query.lower()):
                query_info['is_temporal'] = True
                break
                
        # Check for conditional queries
        conditional_patterns = [
            r'(if|when|where|in case|assuming|provided that|given that)',
            r'(greater than|less than|equal to|more than|fewer than)',
            r'(before|after|during|while)'
        ]
        
        for pattern in conditional_patterns:
            if re.search(pattern, query.lower()):
                query_info['has_condition'] = True
                break
                
        # Check for comparative queries
        if re.search(r'(compare|difference|similarities|versus|vs|compared to|better|worse|faster|slower|higher|lower|more|less)', query.lower()):
            query_info['is_comparative'] = True
            
        # Check for multi-hop questions
        if re.search(r'(who|what|where|when|why|how).+(who|what|where|when|why|how)', query.lower()):
            query_info['is_multi_hop'] = True
            
        # Extract potential entities (proper nouns)
        entity_matches = re.findall(r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b', query)
        # Filter out common words that might be capitalized
        common_words = {"what", "who", "where", "when", "why", "how", "is", "are", "was", "were", "the", "a", "an"}
        query_info['entities'] = [e for e in entity_matches if e.lower() not in common_words]
        
        # Extract keywords (nouns, verbs, adjectives) - simple approach
        words = query.split()
        # Remove stop words and punctuation
        stop_words = {"the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be", "to", "of", "in", "that", "have", "with", "for", "this", "that"}
        query_info['keywords'] = [word.lower().rstrip(",.?!:;") for word in words if word.lower() not in stop_words and len(word) > 2]
        
        # Determine question type
        if re.search(r'\b(list|name|enumerate|identify)\b', query.lower()):
            query_info['question_type'] = 'list'
        elif re.search(r'\b(explain|why|how come|reason|describe|elaborate)\b', query.lower()):
            query_info['question_type'] = 'explanation'
        elif re.search(r'\b(compare|difference|similarities|versus|vs|compared to)\b', query.lower()):
            query_info['question_type'] = 'comparison'
        elif re.search(r'\b(calculate|compute|how many|how much|count|sum|total|average|mean)\b', query.lower()):
            query_info['question_type'] = 'calculation'
            
        # Store in cache
        self.query_cache[cache_key] = query_info
        return query_info

    def get_batch_size(self) -> int:
        """
        Determines the batch size that is used by the evaluator.
        """
        self.batch_size = AICROWD_SUBMISSION_BATCH_SIZE
        return self.batch_size

    def batch_generate_answer(self, batch: Dict[str, Any]) -> List[str]:
        """
        Generates answers for a batch of queries with enhanced RAG capabilities.
        """
        batch_interaction_ids = batch["interaction_id"]
        queries = batch["query"]
        batch_search_results = batch["search_results"]
        query_times = batch["query_time"]

        # Extract chunks with improved chunk extractor
        chunks, chunk_interaction_ids, metadata_dict = self.chunk_extractor.extract_chunks(
            batch_interaction_ids, batch_search_results
        )

        # Parse queries for better understanding
        query_info_list = [self.parse_query(query, query_times[i]) for i, query in enumerate(queries)]

        # Calculate embeddings for chunks and queries
        if len(chunks) > 0:
            chunk_embeddings = self.calculate_embeddings(chunks)
            query_embeddings = self.calculate_embeddings(queries)
        else:
            # Handle the edge case of no chunks
            return ["I don't know"] * len(queries)

        # Retrieve top matches for each query with query-specific processing
        batch_retrieval_results = []
        batch_metadata = []
        
        for _idx, interaction_id in enumerate(batch_interaction_ids):
            query = queries[_idx]
            query_time = query_times[_idx]
            query_embedding = query_embeddings[_idx]
            query_info = query_info_list[_idx]

            # Identify chunks that belong to this interaction_id
            relevant_chunks_mask = chunk_interaction_ids == interaction_id
            
            # If no relevant chunks found, return "I don't know"
            if not np.any(relevant_chunks_mask):
                batch_retrieval_results.append([])
                batch_metadata.append("")
                continue

            # Filter relevant chunks and embeddings
            relevant_chunks = chunks[relevant_chunks_mask]
            relevant_chunks_embeddings = chunk_embeddings[relevant_chunks_mask]

            # Calculate cosine similarity between query and chunk embeddings
            cosine_scores = (relevant_chunks_embeddings * query_embedding).sum(1)
            
            # Apply query-specific boosting based on parsed query information
            if query_info['is_temporal']:
                # Boost chunks containing temporal information
                temporal_patterns = [
                    r'(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)(\s+\d{1,2})?(\s*,\s*\d{4})?',
                    r'\b\d{4}\b',  # Years
                    r'(today|yesterday|current|previous|next|recent)'
                ]
                for i, chunk in enumerate(relevant_chunks):
                    for pattern in temporal_patterns:
                        if re.search(pattern, chunk.lower()):
                            cosine_scores[i] *= 1.2  # Boost score by 20%
                            break
            
            # Boost chunks containing entity names mentioned in the query
            for entity in query_info['entities']:
                for i, chunk in enumerate(relevant_chunks):
                    if entity.lower() in chunk.lower():
                        cosine_scores[i] *= 1.15  # Boost score by 15%
            
            # For comparative questions, try to ensure we get chunks about both entities
            if query_info['is_comparative'] and len(query_info['entities']) >= 2:
                # Track if we have info about each entity
                entity_coverage = {entity.lower(): False for entity in query_info['entities']}
                
                # First, get the top half of our needed chunks based on similarity
                half_count = NUM_CONTEXT_SENTENCES // 2
                initial_indices = (-cosine_scores).argsort()[:half_count]
                selected_indices = list(initial_indices)
                
                # Check which entities are covered in our initial selection
                for idx in initial_indices:
                    chunk = relevant_chunks[idx]
                    for entity in query_info['entities']:
                        if entity.lower() in chunk.lower():
                            entity_coverage[entity.lower()] = True
                
                # For uncovered entities, find the best chunks
                remaining_indices = (-cosine_scores).argsort()[half_count:]
                for entity in query_info['entities']:
                    if not entity_coverage[entity.lower()]:
                        for idx in remaining_indices:
                            if idx not in selected_indices and entity.lower() in relevant_chunks[idx].lower():
                                selected_indices.append(idx)
                                entity_coverage[entity.lower()] = True
                                break
                
                # Fill remaining slots with highest similarity chunks
                while len(selected_indices) < NUM_CONTEXT_SENTENCES and len(remaining_indices) > 0:
                    for idx in remaining_indices:
                        if idx not in selected_indices:
                            selected_indices.append(idx)
                            if len(selected_indices) >= NUM_CONTEXT_SENTENCES:
                                break
                
                # Get retrieval results using selected indices
                retrieval_results = relevant_chunks[selected_indices]
            else:
                # Standard approach: just get top-N results by similarity
                retrieval_results = relevant_chunks[(-cosine_scores).argsort()[:NUM_CONTEXT_SENTENCES]]
            
            # Add metadata
            batch_metadata.append(metadata_dict[interaction_id])
            batch_retrieval_results.append(retrieval_results)
            
        # Prepare formatted prompts for the LLM
        formatted_prompts = self.format_prompts(
            queries,
            query_times,
            batch_retrieval_results,
            batch_metadata,
            query_info_list
        )

        # Generate responses
        if self.is_server:
            response = self.llm_client.chat.completions.create(
                model=self.llm_name,
                messages=formatted_prompts[0],
                n=1,
                top_p=0.9,
                temperature=0.1,
                max_tokens=50,
            )
            answers = [response.choices[0].message.content]
        else:
            responses = self.llm.generate(
                formatted_prompts,
                vllm.SamplingParams(
                    n=1,
                    top_p=0.9,
                    temperature=0.1,
                    skip_special_tokens=True,
                    max_tokens=50,
                ),
                use_tqdm=False
            )
            answers = []
            for response in responses:
                answers.append(response.outputs[0].text)

        # Post-process answers
        processed_answers = []
        for i, answer in enumerate(answers):
            # Clean up the answer
            processed = self.post_process_answer(answer, query_info_list[i])
            processed_answers.append(processed)
            
        return processed_answers

    def post_process_answer(self, answer: str, query_info: Dict) -> str:
        """
        Post-process the generated answer for improved quality.
        
        Args:
            answer: The raw answer from the model
            query_info: The parsed query information
            
        Returns:
            Cleaned and formatted answer
        """
        # Remove explanatory phrases if present
        explanation_patterns = [
            r'Based on the (provided |)references,\s*',
            r'According to the (provided |)references,\s*',
            r'From the (provided |)references,\s*',
            r'The references (indicate|suggest|show) that\s*',
        ]
        
        for pattern in explanation_patterns:
            answer = re.sub(pattern, '', answer, flags=re.IGNORECASE)
        
        # Clean up formatting
        answer = answer.strip()
        
        # Remove starting and trailing quotes if present
        if answer.startswith('"') and answer.endswith('"'):
            answer = answer[1:-1]
            
        # Special handling for list questions
        if query_info['question_type'] == 'list' and ',' in answer and not answer.lower().startswith("i don't know"):
            # Ensure proper comma formatting
            items = [item.strip() for item in re.split(r',\s*', answer)]
            # Remove duplicate items while preserving order
            seen = set()
            unique_items = []
            for item in items:
                if item.lower() not in seen:
                    seen.add(item.lower())
                    unique_items.append(item)
            answer = ", ".join(unique_items)
            
        # Ensure answers don't exceed 50 tokens (approximately 50-75 words)
        words = answer.split()
        if len(words) > 50:
            answer = " ".join(words[:50])
            
        # Final cleanup to ensure consistent formatting
        answer = answer.strip()
        if not answer:
            answer = "I don't know"
            
        return answer

    def format_prompts(self, queries, query_times, batch_retrieval_results=[], batch_metadata=[], query_info_list=[]):
        """
        Format prompts with sophisticated prompt engineering techniques.
        """
        # Base system prompt with clear instructions
        base_system_prompt = """You are a precise question-answering assistant. Your task is to answer questions based ONLY on the provided references.
1. If the references contain the answer, respond with a concise, direct answer using the fewest words possible.
2. If the references don't contain sufficient information to answer the question, respond ONLY with 'I don't know'.
3. NEVER include explanations, reasoning, or information not found in the references.
4. Pay careful attention to dates, numbers, and specific details in the references.
5. For numerical or calculation questions, provide the exact figure from the references.
6. For list questions, provide a simple comma-separated list of items.
7. For comparison questions, ensure your answer addresses both compared items.
8. Keep your response under 50 words and aim for brevity.
"""

        formatted_prompts = []

        for _idx, query in enumerate(queries):
            query_time = query_times[_idx]
            retrieval_results = batch_retrieval_results[_idx] if _idx < len(batch_retrieval_results) else []
            metadata = batch_metadata[_idx] if _idx < len(batch_metadata) else ""
            query_info = query_info_list[_idx] if _idx < len(query_info_list) else {}
            
            # Customize system prompt based on query type
            system_prompt = base_system_prompt
            if query_info.get('question_type') == 'calculation':
                system_prompt += "\nThis question requires numerical calculation. Be precise with numbers and calculations."
            elif query_info.get('question_type') == 'comparison':
                system_prompt += "\nThis question requires comparing multiple items. Be sure to address all items being compared."
            elif query_info.get('question_type') == 'list':
                system_prompt += "\nThis question requires a list of items. Provide a simple comma-separated list of items from the references."
            
            # Add temporal context if relevant
            if query_info.get('is_temporal', False):
                system_prompt += f"\nThe current time when the question was asked is: {query_time}."

            # Format the user message with references
            user_message = ""
            references = ""
            
            if len(retrieval_results) > 0:
                references += "# References\n"
                # Format the retrieved passages with source information if available
                for i, snippet in enumerate(retrieval_results):
                    references += f"{i+1}. {snippet.strip()}\n"
                
                # Add metadata about sources if available
                if metadata:
                    references += f"\n# Source Information\n{metadata}\n"
            
            # Trim references to fit within limits
            references = references[:MAX_CONTEXT_REFERENCES_LENGTH]
            
            # Construct final user message
            user_message += f"{references}\n------\n\n"
            user_message += f"Answer the following question using ONLY the information in the references above.\n"
            user_message += f"Current Time: {query_time}\n"
            user_message += f"Question: {query}\n"

            if self.is_server:
                # Format for OpenAI API
                formatted_prompts.append(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ]
                )
            else:
                # Format for tokenizer
                formatted_prompts.append(
                    self.tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message},
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )

        return formatted_prompts
