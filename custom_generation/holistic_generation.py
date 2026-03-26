import json
import os
from openai import OpenAI
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import model_generate, extract_python_code, ds_api_generate

data_path = "/data0/xjh/ClassEval/data/ClassEval_data.json"
model_path = "/data1/model/qwen/Qwen/Qwen2.5-Coder-1.5B-Instruct/"
output_dir = "/data0/xjh/ClassEval/custom_generation/ds_holistic"

USE_DS_API = True

os.makedirs(output_dir, exist_ok=True)

with open(data_path, "r") as f:
    data = json.load(f)

if USE_DS_API:
    client = OpenAI(api_key=os.environ.get('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")
else:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

def process_single_sample(problem_info: dict):
    class_skeleton = problem_info["skeleton"]
    prompt = "Please implement the following the class.\n"
    prompt += class_skeleton
    prompt += "\nOnly provide the method implementation without any explanation."

    if USE_DS_API:
        generated_text = ds_api_generate(prompt, client)
    else:
        generated_text = model_generate(prompt, model, tokenizer)
    code = extract_python_code(generated_text)

    return code

def main():
    for problem in data:
        task_id = problem["task_id"]
        code = process_single_sample(problem)
        with open(os.path.join(output_dir, f"{task_id}.py") , "w") as f:
            f.write(code)


if __name__ == "__main__":
    main()