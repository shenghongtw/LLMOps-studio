# LLMOps-studio Platform User Guide
[中文文檔](README-TW.md)
## Overview

This project is built on OpenWebUI, providing a one-stop LLM development, training, and application platform that integrates automated training, model management, local model deployment, and RAG applications.


## Quick Start

### Environment Configuration

1. Clone this repository
```bash
git clone https://github.com/yourusername/openwebui.git
cd openwebui
```

2. Configure environment variables
```bash
cp .env.example .env
# Edit the .env file and configure necessary API keys
```

3. Start the service
```bash
docker-compose up -d
```

4. Access the Web interface
http://localhost:3000

## Main Features

### Automated Training Execution Steps

The platform supports automatically generating training samples based on user conversation data and fine-tuning using LLaMA-Factory:

1. The system automatically fetches new conversation records from PostgreSQL
2. Processes conversation data into training sample format
3. Uses LLaMA-Factory for LoRA fine-tuning
4. Merges models and registers them to MLflow
5. Converts to GGUF format and deploys to Ollama service

For detailed training process, see `llmops/workflows/training_deploy_pipeline.py`.

### MLflow Prompt Tuning Experiments
1. Configure content in the prompt_experiments.yaml file
   - Define experiment name, data path, and evaluation model:
     ```yaml
     experiment_name: your_experiment_name
     data_path: llmops/data/your_test_data.json
     judge_model: openai:/gpt-4o
     ```
   
   - Configure models to test and their parameters:
     ```yaml
     models:
       gpt-4o-mini:
         temperature: [0.0, 0.3, 0.7]  # Can specify multiple values for comparison
         top_p: 1.0
         max_tokens: 256
     ```
   
   - Design different prompt templates:
     ```yaml
     prompts:
       template_name:
         template: |
           Your prompt template, can use {{ variable }} syntax to reference variables
         metadata:
           description: Prompt description
           tags: [tag1, tag2]
     ```
2. run experiments
```bash
docker-compose exec -it llmops bash
python gpt-4o-promtp.py
```
3. Access the MLflow interface: `http://localhost:5000`
4. View registered models and experiments
5. Compare performance of different model versions through MLflow UI

### RAG Application Pipeline

The platform integrates a complete RAG (Retrieval Augmented Generation) pipeline:

1. Document upload and processing
2. Pinecone-based vector storage
3. Intelligent Q&A with source document references

For RAG processing workflow, see `pipelines/pipelines/rag.py`.

RAG Execution Steps:
1. Place the PDF documents you want to retrieve into /pipelines/pipelines/data and delete the test.pdf file in it


## System Architecture

The system consists of the following main components:

- **Open-WebUI**: Provides user interface and conversation management
- **Ollama**: Local large model service
- **LLMOps**: Model training and management
- **MLflow**: Experiment tracking and model registration
- **Pipelines**: RAG and other application pipelines
- **PostgreSQL**: Data storage
- **MinIO**: Object storage

## Directory Structure

Key development directories:

### /llmops/experiments
This directory is mainly used for prompt engineering experiments. Currently includes:
- gpt-4o-prompt.py: Script for testing prompts across different models
- configs/: Experiment configuration files
- This area supports various prompt optimizations, comparison of different models' responses to the same prompts, and systematic prompt experiments with results tracked and compared through MLflow.

### /llmops/workflows
Automated training pipelines using Prefect, currently including:
- training_deploy_pipeline.py: Complete training deployment pipeline that automates the entire process from data processing to model training, evaluation, and deployment
- This coordinates the entire model training, evaluation, and deployment process, enabling full automation from data preparation to final model deployment.

### /pipelines/pipelines
This directory contains custom complex AI tasks that can be used through the OpenWebUI interface. Currently implemented:
- rag.py: Pinecone-based vector retrieval and Langchain RAG implementation
- These pipelines can be extended to various complex AI tasks such as RAG (Retrieval Augmented Generation), multi-agent systems, long-text analysis, automated report generation, etc.


