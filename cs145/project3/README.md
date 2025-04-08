# Setup Steps

## 1. Create a conda env:

conda create -n rag python=3.10
conda activate rag

## 2. Install the requirements:

pip install -r requirements.txt
pip install --upgrade openai # make sure you have the latest version

## 3. Login to HuggingFace (you need access to the models):

huggingface-cli login --token "your_token_here"

## 4. Download the dataset and put it in the data/ folder

# Execution Steps

## I used Offline Mode on a Google GCP.

python generate.py \
 --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" \
 --split 1 \
 --model_name "enhanced_rag" \
 --llm_name "meta-llama/Llama-3.2-3B-Instruct"

# System Components

## My enhanced RAG has these main improvements:

- Better chunk extraction - recognizes HTML structure and extracts tables better
- Query understanding - classifies different question types and finds entities
- Smarter retrieval - boosts relevant chunks based on question type
- Better prompting - uses different prompts for different question types
- Post-processing - cleans up responses based on question type

## The main files are:

- enhanced_rag.py - My implementation
- generate.py - For running inference
- evaluate.py - For evaluation
- requirements.txt - Dependencies

# Pipeline Setup

## 1. Query Processing

- The system takes a user query and parses it with the parse_query method
- Identifies question type (factual, list, comparative, etc.)
- Detects temporal aspects (if query mentions dates/time)
- Extracts entities and keywords
- Caches processed queries to avoid redundant work

## 2. Document Processing

- HTML documents are processed by the ChunkExtractor class
- Prioritizes main content using semantic HTML tags
- Extracts tables and converts to text format
- Collects metadata (source URL, publish date)
- Removes boilerplate content and script elements

## 3. Chunking & Embedding

- Documents are split into meaningful chunks
- Advanced deduplication removes similar content while preserving order
- Chunks are embedded using the better all-mpnet-base-v2 model
- Queries are embedded using the same model for consistency

## 4. Retrieval

- System computes similarity between query and chunks
- Applies query-specific boosting:
  - Entity matches get 15% boost
  - Temporal information gets 20% boost
  - For comparative questions, ensures coverage of all compared entities

## 5. Context Assembly & Prompt Engineering

- Formats retrieved chunks as references
- Constructs specialized prompt based on question type
- Includes metadata about sources when available
- Limits context length to fit model constraints

## 6. Generation & Post-processing

- Uses model (Llama-3.2-3B-Instruct) to generate answer
- Post-processes output:
  - Removes explanatory phrases
  - Formats list responses consistently
  - Controls answer length
  - Applies question-type specific formatting
