import mlflow
import pandas as pd
import openai
import itertools
import yaml
import os
from typing import Dict, List
import argparse
from datetime import datetime

# 1. 從YAML文件加載實驗配置
def load_experiment_config(config_path: str) -> Dict:
    """從YAML文件加載實驗配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# 2. 準備評估數據
def prepare_eval_data(data_path: str = None):
    """準備評估數據，可以從文件加載或使用預設"""
    if data_path and os.path.exists(data_path):
        return pd.read_json(data_path)
    
    return pd.DataFrame(
        {
            "inputs": [
                "Artificial intelligence has transformed how businesses operate in the 21st century. Companies are leveraging AI for everything from customer service to supply chain optimization. The technology enables automation of routine tasks, freeing human workers for more creative endeavors. However, concerns about job displacement and ethical implications remain significant. Many experts argue that AI will ultimately create more jobs than it eliminates, though the transition may be challenging.",
                "Climate change continues to affect ecosystems worldwide at an alarming rate. Rising global temperatures have led to more frequent extreme weather events including hurricanes, floods, and wildfires. Polar ice caps are melting faster than predicted, contributing to sea level rise that threatens coastal communities. Scientists warn that without immediate and dramatic reductions in greenhouse gas emissions, many of these changes may become irreversible. International cooperation remains essential but politically challenging.",
                "The human genome project was completed in 2003 after 13 years of international collaborative research. It successfully mapped all of the genes of the human genome, approximately 20,000-25,000 genes in total. The project cost nearly $3 billion but has enabled countless medical advances and spawned new fields like pharmacogenomics. The knowledge gained has dramatically improved our understanding of genetic diseases and opened pathways to personalized medicine. Today, a complete human genome can be sequenced in under a day for about $1,000.",
                "Remote work adoption accelerated dramatically during the COVID-19 pandemic. Organizations that had previously resisted flexible work arrangements were forced to implement digital collaboration tools and virtual workflows. Many companies reported surprising productivity gains, though concerns about company culture and collaboration persisted. After the pandemic, a hybrid model emerged as the preferred approach for many businesses, combining in-office and remote work. This shift has profound implications for urban planning, commercial real estate, and work-life balance.",
                "Quantum computing represents a fundamental shift in computational capability. Unlike classical computers that use bits as either 0 or 1, quantum computers use quantum bits or qubits that can exist in multiple states simultaneously. This property, known as superposition, theoretically allows quantum computers to solve certain problems exponentially faster than classical computers. Major technology companies and governments are investing billions in quantum research. Fields like cryptography, material science, and drug discovery are expected to be revolutionized once quantum computers reach practical scale.",
            ],
            "targets": [
                "AI has revolutionized business operations through automation and optimization, though ethical concerns about job displacement persist alongside predictions that AI will ultimately create more employment opportunities than it eliminates.",
                "Climate change is causing accelerating environmental damage through extreme weather events and melting ice caps, with scientists warning that without immediate reduction in greenhouse gas emissions, many changes may become irreversible.",
                "The Human Genome Project, completed in 2003, mapped approximately 20,000-25,000 human genes at a cost of $3 billion, enabling medical advances, improving understanding of genetic diseases, and establishing the foundation for personalized medicine.",
                "The COVID-19 pandemic forced widespread adoption of remote work, revealing unexpected productivity benefits despite collaboration challenges, and resulting in a hybrid work model that impacts urban planning, real estate, and work-life balance.",
                "Quantum computing uses qubits existing in multiple simultaneous states to potentially solve certain problems exponentially faster than classical computers, with major investment from tech companies and governments anticipating revolutionary applications in cryptography, materials science, and pharmaceutical research.",
            ],
        }
    )

# 3. 註冊提示模板
def register_prompt_templates(prompt_configs: Dict) -> Dict:
    """根據配置註冊提示模板"""
    registered_prompts = {}
    for name, template_config in prompt_configs.items():
        template = template_config["template"]
        prompt_name = f"summarization-prompt-{name}"
        prompt = mlflow.register_prompt(
            name=prompt_name,
            template=template,
            commit_message=f"Register {name} template",
        )
        registered_prompts[name] = {
            "prompt_obj": prompt,
            "metadata": template_config.get("metadata", {})
        }
        print(f"註冊提示模板 '{prompt.name}' (版本 {prompt.version})")
    return registered_prompts

# 4. 預測函數工廠
def create_predict_fn(model_name: str, model_params: Dict, prompt_name: str, prompt_version: int = 1):
    """建立預測函數"""
    def predict_fn(data: pd.DataFrame) -> List[str]:
        predictions = []
        prompt = mlflow.load_prompt(f"prompts:/{prompt_name}/{prompt_version}")

        for _, row in data.iterrows():
            content = prompt.format(sentences=row["inputs"], num_sentences=1)
            completion = openai.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": content}],
                **model_params
            )
            predictions.append(completion.choices[0].message.content)

        return predictions
    
    return predict_fn

# 5. 運行實驗
def run_experiments(config: Dict):
    """運行所有實驗"""
    # 準備數據
    eval_data = prepare_eval_data(config.get("data_path"))
    
    # 註冊提示模板
    registered_prompts = register_prompt_templates(config["prompts"])
    
    results = {}
    experiment_configs = []
    
    # 建立配置矩陣
    for model_name, model_config in config["models"].items():
        model_params_list = []
        
        # 處理每個參數的多個值
        param_keys = []
        param_values = []
        
        for param_name, param_value in model_config.items():
            if isinstance(param_value, list):
                param_keys.append(param_name)
                param_values.append(param_value)
            elif isinstance(param_value, dict) and all(k in param_value for k in ["min", "max"]):
                # 處理參數範圍
                min_val = param_value["min"]
                max_val = param_value["max"]
                step = param_value.get("step", (max_val - min_val) / 4)
                values = [min_val + i * step for i in range(int((max_val - min_val) / step) + 1)]
                param_keys.append(param_name)
                param_values.append(values)
            else:
                # 單一值參數
                pass
        
        # 生成所有參數組合
        if param_values:
            for combo in itertools.product(*param_values):
                params = {k: v for k, v in zip(param_keys, combo)}
                # 添加非列表參數
                for k, v in model_config.items():
                    if not isinstance(v, list) and not isinstance(v, dict):
                        params[k] = v
                model_params_list.append(params)
        else:
            # 沒有多值參數
            model_params_list.append({k: v for k, v in model_config.items() 
                                     if not isinstance(v, dict)})
        
        # 將模型與所有參數組合匹配
        for model_params in model_params_list:
            for prompt_name in config["prompts"].keys():
                experiment_configs.append((model_name, model_params, prompt_name))
    
    # 執行所有實驗
    total_configs = len(experiment_configs)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = config.get(f"experiment_name{timestamp}", f"prompt_experiment_{timestamp}")
    
    print(f"開始執行 {total_configs} 個實驗配置 ({experiment_name})...")
    
    # 創建MLflow實驗
    mlflow.set_experiment(experiment_name)
    
    for i, (model, params, prompt_type) in enumerate(experiment_configs, 1):
        prompt_name = f"summarization-prompt-{prompt_type}"
        
        # 為配置創建一個簡潔的描述
        param_desc = "_".join([f"{k}={v}" for k, v in params.items() 
                              if k in ['temperature', 'top_p', 'max_tokens']])
        run_name = f"{model}-{prompt_type}-{param_desc}"
        
        print(f"[{i}/{total_configs}] 運行實驗: {run_name}")
        
        with mlflow.start_run(run_name=run_name):
            # 記錄參數
            mlflow.log_param("model", model)
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)
            mlflow.log_param("prompt_type", prompt_type)
            
            # 記錄提示模板
            mlflow.log_text(config["prompts"][prompt_type]["template"], "prompt_template.txt")
            
            # 創建預測函數
            predict_fn = create_predict_fn(model, params, prompt_name)
            
            try:
                # 評估模型
                result = mlflow.evaluate(
                    model=predict_fn,
                    data=eval_data,
                    targets="targets",
                    model_type="text-summarization",
                    # extra_metrics=[
                    #     mlflow.metrics.latency(),
                    #     mlflow.metrics.genai.answer_similarity(model=config.get("judge_model", "openai:/gpt-4")),
                    # ],
                )
                
                results[(model, str(params), prompt_type)] = result
                print(f"完成: {run_name}")
            except Exception as e:
                print(f"實驗 {run_name} 失敗: {str(e)}")
                mlflow.log_text(str(e), "error.txt")
    
    return results, experiment_name

# 6. 結果分析
# def analyze_results(results, experiment_name):
#     """分析實驗結果"""
#     if not results:
#         print("沒有成功的實驗結果可供分析")
#         return None
        
#     # 創建比較用的DataFrame
#     comparison = []
    
#     for (model, params_str, prompt_type), result in results.items():
#         metrics = result.metrics
#         row = {
#             "模型": model,
#             "參數": params_str,
#             "提示類型": prompt_type,
#             "相似度平均": metrics.get("answer_similarity_mean", 0),
#             "相似度中位數": metrics.get("answer_similarity_median", 0),
#             "延遲(秒)": metrics.get("latency_mean", 0),
#         }
#         comparison.append(row)
    
#     df = pd.DataFrame(comparison)
    
#     # 找出最佳配置
#     if not df.empty:
#         best_by_similarity = df.loc[df["相似度平均"].idxmax()]
#         best_by_latency = df.loc[df["延遲(秒)"].idxmin()]
        
#         print("\n最佳配置 (按相似度):")
#         print(best_by_similarity)
        
#         print("\n最佳配置 (按延遲):")
#         print(best_by_latency)
        
#         # 將結果保存到CSV
#         output_dir = "results"
#         os.makedirs(output_dir, exist_ok=True)
#         output_file = f"{output_dir}/{experiment_name}_results.csv"
#         df.to_csv(output_file, index=False)
#         print(f"\n結果已保存到 {output_file}")
    
#     return df

# 7. 主執行函數
def main():
    parser = argparse.ArgumentParser(description='運行提示模板實驗')
    parser.add_argument('--config', '-c', type=str, required=True, 
                        help='實驗配置YAML文件路徑')
    args = parser.parse_args()
    
    config = load_experiment_config(args.config)
    results, experiment_name = run_experiments(config)
    # comparison_df = analyze_results(results, experiment_name)
    
    # # 保存配置文件副本到結果目錄
    # if comparison_df is not None:
    #     os.makedirs("results", exist_ok=True)
    #     with open(f"results/{experiment_name}_config.yaml", "w", encoding="utf-8") as f:
    #         yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

if __name__ == "__main__":
    main()