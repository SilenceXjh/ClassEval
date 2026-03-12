import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

data_path = "/data0/xjh/ClassEval/data/ClassEval_data.json"
model_path = "/data1/model/qwen/Qwen/Qwen2.5-Coder-7B-Instruct/"

with open(data_path, "r") as f:
    data = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

def get_sorted_methods(method_info_dict: dict):
    n = len(method_info_dict)
    sorted_methods = []
    for _ in range(n):
        for method_name in method_info_dict.keys():
            if method_name in sorted_methods:
                continue
            dependencies = method_info_dict[method_name]["dependencies"]
            depend_methods = dependencies["method_dependencies"]
            has_depend = False
            for depend_m in depend_methods:
                if depend_m not in sorted_methods:
                    has_depend = True
                    break
            if not has_depend:
                sorted_methods.append(method_name)
                break
    return sorted_methods


def add_desc_to_init(desc, class_init):
        class_init_list = class_init.split('\n')
        class_init_list[0] += " \n" + desc
        class_init = '\n'.join(class_init_list)
        return class_init


def model_generate(prompt: str):
    messages = [
        {"role": "system", "content": "You are an expert Python programmer."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    # print("model input text:", text)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512, 
            temperature=0.2,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if prompt in generated_text:
        generated_text = generated_text.split(prompt)[-1].strip()
    
    print("model generated text:", generated_text)
    return generated_text

def extract_python_code(generated_text: str):
    if "```python" in generated_text:
        code = generated_text.split("```python")[1].split("```")[0].strip()
    elif "```" in generated_text:
        code = generated_text.split("```")[1].split("```")[0].strip()
    else:
        code = generated_text.strip()

    return code


def process_single_sample(problem_info: dict):
    class_name = problem_info["class_name"]
    method_info_list = problem_info["methods_info"]
    method_info_dict = {}
    for method_info in method_info_list:
        method_name = method_info["method_name"]
        method_info_dict[method_name] = method_info

    sorted_methods = get_sorted_methods(method_info_dict)
    # print("sorted methods:", sorted_methods)

    imports = '\n'.join(problem_info['import_statement'])
    class_init = add_desc_to_init(problem_info['class_description'], problem_info['class_constructor'])
    class_text = imports + '\n' + class_init
    # print("class text:")
    # print(class_text)

    for method_name in sorted_methods:
        method_info = method_info_dict[method_name]
        class_text_desc = class_text + "\n\n    " + method_info['method_description']
        test_code = method_info["test_code"]
        # print("class_text_desc:")
        # print(class_text_desc)
        prompt = f"Please complete {method_name} method in the following class {class_name}\n\n"
        prompt += class_text_desc + "\n\n"
        prompt += "You implementation should pass the following tests:\n"
        prompt += test_code + "\n\n"
        prompt += "Only provide the method implementation without any explanation."
        # print("prompt:")
        # print(prompt)

        generated_text = model_generate(prompt)
        code = extract_python_code(generated_text)
        # print("code:")
        # print(code)
        code_lines = code.split("\n")
        class_text += "\n"
        for code_line in code_lines:
            class_text += "\n    " + code_line

    return class_text


def main():
    for problem in data:
        task_id = problem["task_id"]
        final_class_text = process_single_sample(problem)
        # print("final class:")
        # print(final_class_text)
        with open(f"{task_id}.py", "w") as f:
            f.write(final_class_text)


if __name__ == "__main__":
    main()