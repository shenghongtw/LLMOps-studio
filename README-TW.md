# LLMOps-studio 平台使用指南

## 概述

本專案基於OpenWebUI構建，提供一站式LLM開發訓練及應用平台，整合了自動訓練、模型管理、本地模型部署和RAG應用等功能。

## 快速開始

### 環境配置

1. 複製本專案
```bash
git clone https://github.com/shenghongtw/LLMOpsStudio
cd LLMOpsStudio
```

2. 配置環境變數
```bash
cp .env.example .env
# 編輯.env檔案，配置必要的API密鑰
```

3. 啟動服務
```bash
docker-compose up -d
```

4. 訪問Web介面
http://localhost:3000

## 主要功能

### 自動訓練執行步驟

本平台支援基於用戶對話數據自動生成訓練樣本，並使用LLaMA-Factory進行微調訓練：

1. 系統自動從PostgreSQL抓取新的對話記錄
2. 處理對話數據為訓練樣本格式
3. 使用LLaMA-Factory進行LoRA微調
4. 合併模型並註冊到MLflow
5. 轉換為GGUF格式並部署到Ollama服務

詳細的訓練流程見`llmops/workflows/training_deploy_pipeline.py`。

執行自動訓練部署管線
```bash
docker-compose exec -it llmops bash
python llmops/workflows/training_deploy_pipeline.py
```

### MLflow提示詞調優實驗

1. 設定prompt_experiments.yaml檔中的內容
   - 定義實驗名稱、數據路徑和評估模型：
     ```yaml
     experiment_name: your_experiment_name
     data_path: llmops/data/your_test_data.json
     judge_model: openai:/gpt-4o
     ```
   
   - 配置要測試的模型及其參數：
     ```yaml
     models:
       gpt-4o-mini:
         temperature: [0.0, 0.3, 0.7]  # 可以指定多個值進行對比
         top_p: 1.0
         max_tokens: 256
     ```
   
   - 設計不同的提示詞模板：
     ```yaml
     prompts:
       template_name:
         template: |
           您的提示詞模板，可使用 {{ variable }} 語法引用變數
         metadata:
           description: 提示詞描述
           tags: [tag1, tag2]
     ```

2. 執行實驗
```bash
docker-compose exec -it llmops bash
python gpt-4o-promtp.py
```

3. 訪問MLflow介面：`http://localhost:5000`
4. 查看已註冊的提示詞和實驗
5. 通過MLflow UI比較不同版本效能

### RAG應用管線

平台整合了完整的RAG(檢索增強生成)管線：

1. 文檔上傳與處理
2. 基於Pinecone的向量儲存
3. 智能問答與源文檔引用

RAG處理流程詳見`pipelines/pipelines/rag.py`。

RAG執行步驟
1.將想要檢索的pdf文檔放入/pipelines/pipelines/data中,並刪除當中的test.pdf


## 系統架構

系統由以下主要組件構成：

- **Open-WebUI**: 提供用戶介面和對話管理
- **Ollama**: 本地大模型服務
- **LLMOps**: 模型訓練與管理
- **MLflow**: 實驗追蹤與模型註冊
- **Pipelines**: 自定義的RAG和其他應用任務
- **PostgreSQL**: 數據儲存
- **MinIO**: 物件儲存

## 資料夾架構

主要開發資料夾說明：

### /llmops/experiments
這個目錄主要用於進行提示詞工程(prompt engineering)的實驗。目前包含：
- gpt-4o-prompt.py：針對不同模型的提示詞實驗腳本
- configs/：實驗配置文件
- 這裡可以進行各種提示詞調優、比較不同模型對相同提示詞的響應，以及進行系統化的提示詞實驗，最終結果可以通過MLflow進行追蹤和比較。

### /llmops/workflows
主要使用prefect進行自動化訓練的pipeline, 目前包含：
- training_deploy_pipeline.py：完整的訓練部署流水線，從數據處理到模型訓練、評估、部署的全流程自動化
- 這裡負責協調整個模型訓練、評估和部署的流程，實現數據從準備到最終模型部署的全自動化。

### /pipelines/pipelines
這個目錄包含了自定義的複雜AI任務，可以通過OpenWebUI介面使用。當前主要實現了：
- rag.py ：基於Pinecone的向量檢索和Langchain的RAG
- 這裡的管線可以擴展為多種複雜AI任務，如RAG (檢索增強生成) 實現、多智能體 (multi agent)、長文本分析、自動化報告生成等。