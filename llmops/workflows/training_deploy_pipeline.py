import os
import subprocess
import requests
import json
import shutil
from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine, text
import mlflow
from transformers import AutoModelForCausalLM, AutoTokenizer
from prefect import flow, task



@task
def train_model(training_config_path: str, llama_factory_dir: str = "/app/LLaMA-Factory"):
    """使用 llama-factory 訓練模型"""
    # 先检查配置文件是否存在
    if not os.path.exists(training_config_path):
        print(f"错误：训练配置文件 {training_config_path} 不存在")
        return False
        
    # 显示配置文件内容以便调试
    try:
        with open(training_config_path, 'r') as f:
            print(f"训练配置文件内容: {f.read()}")
    except Exception as e:
        print(f"读取配置文件出错: {str(e)}")
    
    # 确保llama-factory目录存在
    if not os.path.exists(llama_factory_dir):
        print(f"错误：llama-factory目录 {llama_factory_dir} 不存在")
        return False
    
    # 切换到llama-factory目录并执行训练命令
    cmd = [
        f"cd {llama_factory_dir}",
        f"llamafactory-cli train {training_config_path}"
    ]

    try:
        process = subprocess.run(
            " && ".join(cmd),
            shell=True,
            check=True,
            text=True,
            capture_output=True
        )
        
        print(process.stdout)
        if process.stderr:
            print("警告或錯誤信息：", process.stderr)
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"训练命令执行失败，退出代码：{e.returncode}")
        print(f"标准输出：{e.stdout}")
        print(f"错误输出：{e.stderr}")
        return False

@task
def merge_model(config_path: str, llama_factory_dir: str = "/app/LLaMA-Factory"):
    """使用 llama-factory 合併模型"""
    # 确保llama-factory目录存在
    if not os.path.exists(llama_factory_dir):
        print(f"错误：llama-factory目录 {llama_factory_dir} 不存在")
        return False
    
    # 切换到llama-factory目录并执行合并命令
    cmd = [
        f"cd {llama_factory_dir}",
        f"llamafactory-cli export {config_path}"
    ]
    
    try:
        process = subprocess.run(
            " && ".join(cmd),
            shell=True,
            check=True,
            text=True,
            capture_output=True
        )
        
        print(process.stdout)
        if process.stderr:
            print("警告或錯誤信息：", process.stderr)
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"合并命令执行失败，退出代码：{e.returncode}")
        print(f"标准输出：{e.stdout}")
        print(f"错误输出：{e.stderr}")
        return False

@task
def register_model_mlflow(model_path, model_name, model_type="hf"):
    """註冊模型到 MLflow"""
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
    
    try:
        with mlflow.start_run(run_name=f"{model_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}") as run:
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("model_type", model_type)
            
            if model_type == "hf":
                # 对于HF模型，首先加载模型和分词器
                try:
                    model = AutoModelForCausalLM.from_pretrained(model_path)
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                    
                    # 明确指定task为"text-generation"
                    model_info = mlflow.transformers.log_model(
                        transformers_model={"model": model, "tokenizer": tokenizer},
                        artifact_path="model",
                        registered_model_name=model_name,
                        task="text-generation"
                    )
                    model_uri = model_info.model_uri
                except Exception as e:
                    print(f"加载Transformers模型出错: {str(e)}")
                    # 使用file:// URI格式来避免S3等远程存储
                    artifact_path = mlflow.log_artifact(model_path, "model")
                    model_uri = f"runs:/{run.info.run_id}/model"
                    model_info = mlflow.register_model(model_uri, model_name)
                    print(f"通过路径註冊 HF 模型 '{model_name}' 版本 {model_info.version}")
                    
            elif model_type == "gguf":
                # 对于GGUF模型，先记录文件，再注册模型
                artifact_path = mlflow.log_artifact(model_path, "model")
                model_uri = f"runs:/{run.info.run_id}/model"
                model_info = mlflow.register_model(model_uri, model_name)
                print(f"成功註冊 GGUF 模型 '{model_name}' 版本 {model_info.version}")
            
            # # 可選：設置 latest 別名
            # try:
            #     client = mlflow.tracking.MlflowClient()
            #     client.set_registered_model_alias(model_name, "latest", model_info.version)
            #     print(f"已設置 '{model_name}@latest' 別名指向版本 {model_info.version}")
            # except Exception as e:
            #     print(f"設置別名失敗: {str(e)}")
            
            return model_uri  # 只返回 URI，不返回版本號
    except Exception as e:
        print(f"註冊模型时发生错误: {str(e)}")
        raise



@task
def convert_to_gguf(hf_model_path, gguf_output_path):
    """將 Hugging Face 模型轉換為 GGUF 格式以用於 llama.cpp"""
    os.makedirs(os.path.dirname(gguf_output_path), exist_ok=True)
    
    # 這裡使用 llama.cpp 的轉換工具
    cmd = [
        "python", "llama-cpp/convert_hf_to_gguf.py",
        hf_model_path,
        "--outfile", gguf_output_path,
        "--outtype", "f16"
    ]
    
    subprocess.run(cmd, check=True)
    return gguf_output_path

@task
def fetch_model_from_mlflow(model_name, version_or_alias="latest"):
    """從 MLflow 獲取模型，支持版本號或別名"""
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
    client = mlflow.tracking.MlflowClient()
    
    if version_or_alias == "latest":
        try:
            # 優先嘗試使用 "latest" 別名
            model_info = client.get_model_version_by_alias(model_name, "latest")
            model_uri = f"models:/{model_name}@latest"
            print(f"通過別名獲取模型: {model_uri}")
        except Exception as e:
            # 如果別名不存在，獲取最新版本
            print(f"別名不存在，嘗試獲取最新版本: {str(e)}")
            latest_version = client.get_latest_versions(model_name)[0].version
            model_uri = f"models:/{model_name}/{latest_version}"
            print(f"獲取模型最新版本: {model_uri}")
    elif version_or_alias.isdigit():
        # 如果是數字，視為版本號
        model_uri = f"models:/{model_name}/{version_or_alias}"
        print(f"通過版本號獲取模型: {model_uri}")
    else:
        # 否則視為別名
        model_uri = f"models:/{model_name}@{version_or_alias}"
        print(f"通過別名獲取模型: {model_uri}")
    
    local_path = mlflow.artifacts.download_artifacts(model_uri)
    return local_path

@task
def generate_modelfile_content(model_path, parameters=None):
    """生成正确格式的Modelfile内容"""
    print(f"开始生成Modelfile内容，模型路径: {model_path}")
    
    if parameters is None:
        parameters = {
            "temperature": 0.7,
            "top_p": 0.9
        }
    
    try:
        # 查找GGUF文件
        gguf_file = None
        gguf_file_path = None
        
        for file in os.listdir(model_path):
            if file.endswith('.gguf'):
                gguf_file = file
                gguf_file_path = os.path.join(model_path, file)
                break
                
        if not gguf_file:
            return {
                "status": "failed",
                "error": "找不到GGUF文件"
            }
        
        print(f"找到GGUF文件: {gguf_file}")
        
        # 创建共享目录路径 - 使用相对路径
        shared_dir = "./models/shared"  # 修改为相对路径
        os.makedirs(shared_dir, exist_ok=True)
        
        # 复制GGUF文件到共享目录
        shared_gguf_path = os.path.join(shared_dir, gguf_file)
        shutil.copy2(gguf_file_path, shared_gguf_path)
        print(f"已复制GGUF文件到共享路径: {shared_gguf_path}")
        
        # 生成Modelfile内容
        modelfile_content = f"FROM /models/shared/{gguf_file}\n\n"
        modelfile_content += "# 模型参数设置\n"
        for param, value in parameters.items():
            modelfile_content += f"PARAMETER {param} {value}\n"
        
        # 添加系统提示词
        modelfile_content += "\n# 系统提示词设置\n"
        modelfile_content += 'SYSTEM "你是一个有帮助的AI助手"\n'
        
        # 写入Modelfile到共享目录
        modelfile_path = os.path.join(shared_dir, f"Modelfile")
        with open(modelfile_path, "w", encoding="utf-8") as f:
            f.write(modelfile_content)
        
        print(f"已将Modelfile写入到: {modelfile_path}")
        print(f"Modelfile内容:\n{modelfile_content}")
        
        return {
            "status": "success",
            "modelfile_content": modelfile_content,
            "modelfile_path": modelfile_path,
            "gguf_file_path": shared_gguf_path,
            "gguf_file": gguf_file,
            "parameters": parameters
        }
            
    except Exception as e:
        print(f"生成Modelfile时发生错误: {str(e)}")
        return {
            "status": "failed",
            "error": str(e)
        }

@task
def deploy_to_ollama_api(gguf_file_path, modelfile_path, model_name, parameters=None, ollama_url="http://ollama:11434"):
    """使用Ollama API部署模型"""
    print(f"开始通过API部署模型到Ollama: {model_name}")
    print(f"gguf_file_path: {gguf_file_path}")
    print(f"modelfile_path: {modelfile_path}")
    
    if parameters is None:
        parameters = {
            "temperature": 0.7,
            "top_p": 0.9
        }
    
    try:
        # 首先检查文件是否存在
        if not os.path.exists(modelfile_path):
            print(f"错误: Modelfile不存在于路径: {modelfile_path}")
            return {
                "status": "failed",
                "error": f"Modelfile不存在: {modelfile_path}"
            }
            
        # 正确的API请求格式
        payload = {
            "name": model_name,
            "path": modelfile_path
        }
        print(f"API请求URL: {ollama_url}/api/create")
        print(f"请求数据: {payload}")
        
        # 发送API请求
        response = requests.post(f"{ollama_url}/api/create", json=payload)
        
        print(f"API响应状态码: {response.status_code}")
        print(f"API响应内容: {response.text}")
        
        if response.status_code == 200:
            print(f"成功通过ollama api创建模型: {model_name}")
            return {
                "status": "success",
                "model_name": model_name,
                "api_response": response.text
            }
        else:
            error_msg = f"API错误: {response.status_code} - {response.text}"
            print(error_msg)
            
            # 尝试备选格式 - 这个格式已经可以工作
            print("尝试使用备选API格式...")
            alt_payload = {
                "model": model_name,
                "files": {"model": gguf_file_path},
                "parameters": parameters
            }
            
            alt_response = requests.post(f"{ollama_url}/api/create", json=alt_payload)
            
            if alt_response.status_code == 200:
                print(f"使用备选格式成功创建模型: {model_name}")
                return {
                    "status": "success",
                    "model_name": model_name,
                    "api_response": alt_response.text
                }
            
            return {
                "status": "failed",
                "model_name": model_name,
                "error": error_msg
            }
    except Exception as e:
        print(f"通过ollama api创建模型失败: {str(e)}")
        return {
            "status": "failed",
            "model_name": model_name,
            "error": str(e)
        }

@flow
def export_model_to_ollama(
    model_name: str,
    version_or_alias: str = "latest",
    parameters: dict = None,
    ollama_name: str = None,
    ollama_url: str = "http://ollama:11434"
):
    """將註冊的模型導出到 Ollama"""
    
    # 如果没有指定 Ollama 名称，使用模型名称
    if ollama_name is None:
        ollama_name = model_name
    
    # 1. 从 MLflow 获取模型
    model_path = fetch_model_from_mlflow(
        model_name=model_name,
        version_or_alias=version_or_alias
    )
    
    # 2. 生成Modelfile内容(独立任务)
    modelfile_result = generate_modelfile_content(
        model_path=model_path,
        parameters=parameters
    )
    
    if modelfile_result.get("status") != "success":
        return {
            "status": "failed",
            "model_name": model_name,
            "message": f"生成Modelfile内容失败: {modelfile_result.get('error', '未知错误')}"
        }
    
    # 3. 使用API部署到Ollama
    deployment_result = deploy_to_ollama_api(
        gguf_file_path=modelfile_result["gguf_file_path"],
        model_name=ollama_name,
        modelfile_path=modelfile_result["modelfile_path"],
        parameters=modelfile_result["parameters"],
        ollama_url=ollama_url
    )

    if deployment_result.get("status") == "success":
        return {
            "status": "success",
            "model_name": model_name,
            "ollama_name": ollama_name,
            "version_or_alias": version_or_alias,
            "method": deployment_result.get("method", "api"),
            "message": f"模型 {model_name} (版本/别名: {version_or_alias}) 已成功导入 Ollama 为 {ollama_name}"
        }
    else:

        return {
            "status": "failed",
            "model_name": model_name,
            "message": f"模型 {model_name} 导入 Ollama 失败: {deployment_result.get('error', '未知错误')}"
        }

@task
def fetch_chat_data_from_postgres():
    """直接從PostgreSQL數據庫獲取所有聊天數據"""
    
    db_user = os.environ.get("POSTGRES_USER", "openwebui")
    db_password = os.environ.get("POSTGRES_PASSWORD", "openwebui_password")
    db_name = os.environ.get("POSTGRES_DB", "openwebui")
    db_host = "postgres"  # 使用容器名稱
    db_port = "5432"
    
    # 構建數據庫連接URL
    db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    
    # 創建數據庫引擎
    engine = create_engine(db_url)
    
    try:
        # 方式1: 使用SQLAlchemy執行SQL查詢
        with engine.connect() as connection:
            result = connection.execute(text("SELECT * FROM chat ORDER BY updated_at DESC"))
            
            chat_data = []
            for row in result:
                # 將行轉換為字典
                chat_dict = {column: value for column, value in zip(result.keys(), row)}
                
                # 處理JSON字段
                if isinstance(chat_dict.get('chat'), dict):
                    # PostgreSQL可能已經解析了JSON
                    pass
                elif isinstance(chat_dict.get('chat'), str):
                    try:
                        chat_dict['chat'] = json.loads(chat_dict['chat'])
                    except:
                        pass
                
                if isinstance(chat_dict.get('meta'), dict):
                    pass
                elif isinstance(chat_dict.get('meta'), str):
                    try:
                        chat_dict['meta'] = json.loads(chat_dict['meta'])
                    except:
                        pass
                
                chat_data.append(chat_dict)
        
        print(f"從PostgreSQL獲取了 {len(chat_data)} 條聊天記錄")
        return chat_data
        
    except Exception as e:
        print(f"從數據庫獲取數據時出錯: {str(e)}")
        raise

@task
def get_last_processed_timestamp(data_dir):
    """獲取最後處理的時間戳，用於增量處理"""
    try:
        # 查找目錄中所有訓練數據文件
        files = list(Path(data_dir).glob("training-data-*.json"))
        print(f"找到的訓練數據文件: {files}")
        if not files:
            return 0
            
        # 從文件名提取時間戳
        timestamps = []
        for file in files:
            # 文件名格式: training-data-1234567890.json
            try:
                ts = int(file.stem.split('-')[-1])
                timestamps.append(ts)
            except ValueError:
                continue
                
        return max(timestamps) if timestamps else 0
    except Exception as e:
        print(f"獲取最後處理時間戳錯誤: {str(e)}")
        return 0

@task
def process_chat_to_training_format(chat_data, last_processed_timestamp):
    """將聊天數據處理成訓練格式
    
    對於每個聊天，產生多個訓練樣本，每個用戶問題作為instruction和input，
    助手回覆作為output
    """
    training_samples = []
    
    # 按更新時間排序聊天數據
    sorted_chats = sorted(chat_data, key=lambda x: x.get("updated_at", 0))
    
    # 保留最新處理的時間戳，用於命名文件
    newest_timestamp = last_processed_timestamp
    
    for chat in sorted_chats:
        # 跳過已處理的聊天數據
        if chat.get("updated_at", 0) <= last_processed_timestamp:
            continue
            
        # 更新最新時間戳
        newest_timestamp = max(newest_timestamp, chat.get("updated_at", 0))
        
        # 處理消息
        messages = chat.get("chat", {}).get("messages", [])
        
        # 確保消息按對話順序排列
        if messages and isinstance(messages, list):
            i = 0
            while i < len(messages) - 1:
                # 檢查是否為用戶-助手對話模式
                if (messages[i].get("role") == "user" and 
                    messages[i+1].get("role") == "assistant"):
                    
                    user_msg = messages[i].get("content", "")
                    assistant_msg = messages[i+1].get("content", "")
                    
                    # 創建訓練樣本
                    sample = {
                        "instruction": "回答以下問題",
                        "input": user_msg,
                        "output": assistant_msg
                    }
                    training_samples.append(sample)
                    
                    i += 2  # 跳到下一個用戶-助手對
                else:
                    i += 1  # 單獨移動一步

    
    return training_samples, newest_timestamp

@task
def save_training_data(training_samples, data_dir, timestamp):
    """儲存訓練數據到JSON文件"""
    if not training_samples:
        print("沒有新的訓練數據需要保存")
        return None
        
    os.makedirs(data_dir, exist_ok=True)
    print(f"訓練數據保存路徑: {data_dir}")
    training_data_file_name = f"training-data-{timestamp}"
    training_data_file_path = os.path.join(data_dir, f"{training_data_file_name}.json")
    
    with open(training_data_file_path, 'w', encoding='utf-8') as f:
        json.dump(training_samples, f, ensure_ascii=False, indent=2)
            
    print(f"已儲存 {len(training_samples)} 筆訓練數據到 {training_data_file_path}")
    return training_data_file_name, training_data_file_path

@task
def update_dataset_info(training_data_file_name):
    """更新dataset_info.json文件，添加训练数据信息"""
    dataset_info_path = "/app/LLaMA-Factory/data/dataset_info.json"
    
    try:
        # 读取现有的dataset_info.json文件
        if os.path.exists(dataset_info_path):
            with open(dataset_info_path, 'r', encoding='utf-8') as f:
                dataset_info = json.load(f)
        else:
            dataset_info = {}

        dataset_info[f"{training_data_file_name}"] = {
            "file_name": f"{training_data_file_name}.json"
        }
        
        # 保存更新后的dataset_info.json
        with open(dataset_info_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
            
        print(f"已更新 {dataset_info_path} 文件，添加了 '{training_data_file_name}' 条目")
        return True
    except Exception as e:
        print(f"更新dataset_info.json文件时出错: {str(e)}")
        return False

@flow(name="聊天數據處理流程")
def process_chat_data_flow(
    data_dir: str = "/app/LLaMA-Factory/data"
):
    """從PostgreSQL處理聊天數據流程"""
    
    # 1. 獲取最後處理的時間戳
    last_timestamp = get_last_processed_timestamp(data_dir)
    print(f"上次處理時間戳: {last_timestamp} ({datetime.fromtimestamp(last_timestamp)})")
    
    # 2. 直接從PostgreSQL數據庫獲取聊天數據
    chat_data = fetch_chat_data_from_postgres()
    print(f"獲取了 {len(chat_data)} 個聊天記錄")
    
    # 3. 處理數據為訓練格式
    training_samples, newest_timestamp = process_chat_to_training_format(
        chat_data, last_timestamp
    )
    print(f"處理了 {len(training_samples)} 個訓練樣本，最新時間戳: {newest_timestamp}")
    
    # 4. 儲存訓練數據
    result = save_training_data(training_samples, data_dir, newest_timestamp)
    if result is None:
        # 處理沒有新訓練數據的情況
        return {
            "processed_count": 0,
            "training_data_file_name": None,
            "timestamp": newest_timestamp,
        }
    else:
        training_data_file_name, training_data_file_path = result
        # 5. 更新dataset_info.json文件
        update_success = update_dataset_info(training_data_file_name)
    
    return {
        "processed_count": len(training_samples),
        "training_data_file_name": training_data_file_name,
        "timestamp": newest_timestamp,
    }

@flow()
def llm_training_deploy_pipeline(
    model_name: str = "Qwen/qwen2.5-0.5b-Instruct",
    deploy_to_ollama: bool = True,
    ollama_parameters: dict = None,
    llama_factory_dir: str = "/app/LLaMA-Factory",
    ollama_url: str = "http://ollama:11434",
    data_dir: str = "/app/LLaMA-Factory/data",
    process_chat_data: bool = True
):
    """完整的 LLM 訓練、註冊和部署流程，包括聊天數據處理"""
    
    timestamp = datetime.now().strftime('%Y%m%d')
    results = {}
    
    # 可選的聊天數據處理步驟
    if process_chat_data:
        chat_result = process_chat_data_flow(
            data_dir=data_dir
        )
        results["chat_processing"] = chat_result
        
        # 檢查是否有新的訓練數據
        if not chat_result.get("training_data_file_name"):
            print("沒有新的訓練數據，終止流程")
            return {
                "status": "skipped",
                "message": "沒有新的訓練數據，跳過模型訓練和部署",
                "timestamp": timestamp
            }
    
    # 定义输出路径
    merged_model_dir = f"/app/models/hg/{model_name}_lora_merged"
    gguf_output_path = f"/app/models/gguf/{model_name}-{timestamp}.gguf"
    
    args = dict(
        stage="sft",                                               # do supervised fine-tuning
        do_train=True,
        model_name_or_path=f"{model_name}",           # use Qwen2.5-0.5B-Instruct model
        dataset=f"{results['chat_processing']['training_data_file_name']}",
        template="qwen",                                           # use qwen prompt template
        finetuning_type="lora",                                    # use LoRA adapters to save memory
        lora_target="all",                                         # attach LoRA adapters to all linear layers
        output_dir=f"/app/models/hg/{model_name}_lora",             # the path to save LoRA adapters
        per_device_train_batch_size=1,                             # the micro batch size
        gradient_accumulation_steps=1,                             # the gradient accumulation steps
        lr_scheduler_type="cosine",                                # use cosine learning rate scheduler
        logging_steps=1,                                           # log every step
        warmup_ratio=0.1,                                          # use warmup scheduler
        save_steps=1,                                              # save checkpoint every step
        learning_rate=5e-5,                                        # the learning rate
        num_train_epochs=1.0,                                      # the epochs of training
        max_samples=500,                                           # use 500 examples in dataset
        max_grad_norm=1.0,                                         # clip gradient norm to 1.0
        loraplus_lr_ratio=16.0,                                    # use LoRA+ algorithm with lambda=16.0
        fp16=True,                                                 # use float16 mixed precision training
        report_to="none",                                          # disable wandb logging
    )
    json.dump(args, open(f"/app/train_{model_name}.json", "w", encoding="utf-8"), indent=2)
    # 1. 训练模型
    train_success = train_model(
        training_config_path=f"/app/configs/train_{model_name}.json",
        llama_factory_dir=llama_factory_dir
    )
    
    if not train_success:
        return {"status": "failed", "message": "模型训练失败", **results}
    
    # 2. 合并模型
    merge_success = merge_model(
        config_path=f"/app/merge_{model_name}.json",
        llama_factory_dir=llama_factory_dir
    )
    
    if not merge_success:
        return {"status": "failed", "message": "模型合并失败", **results}
    
    # 3. 转换为 GGUF 格式
    gguf_model_path = convert_to_gguf(
        merged_model_dir,
        gguf_output_path
    )
    
    # 4. 注册原始 HF 模型到 MLflow
    hf_model_uri = register_model_mlflow(
        merged_model_dir,
        f"{model_name}-hf",
        model_type="hf"
    )
    
    # 5. 注册 GGUF 模型到 MLflow
    gguf_model_uri = register_model_mlflow(
        gguf_model_path, 
        f"{model_name}-gguf", 
        model_type="gguf"
    )
    
    results.update({
        "status": "success",
        "hf_model_uri": hf_model_uri,
        "gguf_model_uri": gguf_model_uri
    })
    
    # 6. 如果需要，部署到 Ollama
    if deploy_to_ollama:
        try:
            ollama_result = export_model_to_ollama(
                model_name=f"{model_name}-gguf",
                version_or_alias="latest",
                parameters=ollama_parameters,
                ollama_name=model_name,
                ollama_url=ollama_url
            )
            results["ollama_deployment"] = ollama_result
        except Exception as e:
            results["ollama_deployment"] = {
                "status": "failed",
                "error": str(e)
            }
    
    return results

if __name__ == "__main__":
    
    llm_training_deploy_pipeline(
        model_name="Qwen/qwen2.5-0.5b-Instruct",
        process_chat_data=True
    )