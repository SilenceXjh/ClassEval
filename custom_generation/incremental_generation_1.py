import json
import os
from openai import OpenAI
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import model_generate, extract_python_code, ds_api_generate

data_path = "/data0/xjh/ClassEval/data/ClassEval_data.json"
model_path = "/data1/model/qwen/Qwen/Qwen2.5-Coder-7B-Instruct/"
output_dir = "/data0/xjh/ClassEval/custom_generation/ds_incremental_1"

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


def reformat_method_desc(method_desc: str):
    lines = method_desc.split('\n')
    lines_with_tab = []
    def_exist = False
    for line in lines:
        line = line.strip()
        if not def_exist:
            if line.startswith("def"):
                def_exist = True
            lines_with_tab.append("    " + line.strip())
        else:
            lines_with_tab.append("        " + line.strip())
    # 重新拼接成字符串
    result = '\n'.join(lines_with_tab)
    
    return result


def extract_pure_code(function_code: str) -> str:
    """
    从函数代码字符串中提取去掉def行和文档字符串后的核心代码
    
    Args:
        function_code: 完整的函数代码字符串
        
    Returns:
        去除def行和文档字符串后的核心代码块（保留缩进格式）
    """
    # 将代码按行分割，并去除每行首尾的换行符
    lines = [line.rstrip('\n') for line in function_code.splitlines()]
    
    # 去除首尾空白行
    while lines and lines[0].strip() == '':
        lines.pop(0)
    while lines and lines[-1].strip() == '':
        lines.pop()
    
    if not lines:
        return ""
    
    # 跳过def行
    def_line_no = -1
    for i in range(0, len(lines)):
        if lines[i].strip().startswith("def"):
            def_line_no = i
            break
    assert def_line_no != -1
    if lines[def_line_no].startswith("def"):
        need_indent = True
    else:
        need_indent = False
    
    # 标记是否在文档字符串中
    in_docstring = False
    docstring_start = '"""'
    result_lines = []
    
    for line in lines[i+1:]:
        stripped_line = line.strip()
        
        # 处理文档字符串的开始/结束
        if not in_docstring and stripped_line.startswith(docstring_start):
            in_docstring = True
            # 检查是否是单行文档字符串（"""开头且"""结尾）
            if stripped_line.endswith(docstring_start) and len(stripped_line) > 3:
                in_docstring = False
            continue
        
        if in_docstring:
            # 检查文档字符串结束
            if docstring_start in stripped_line:
                in_docstring = False
            continue
        
        # 收集有效代码行
        if need_indent:
            result_lines.append("    " + line)
        else:
            result_lines.append(line)
        
    
    # 去除结果中首尾的空白行
    while result_lines and result_lines[0].strip() == '':
        result_lines.pop(0)
    while result_lines and result_lines[-1].strip() == '':
        result_lines.pop()
    
    # 拼接成最终的代码块
    return '\n'.join(result_lines)



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
        method_desc = method_info['method_description']
        method_desc = reformat_method_desc(method_desc)
        class_text_desc = class_text + "\n\n" + method_desc
        # print("class_text_desc:")
        # print(class_text_desc)
        prompt = f"Please complete {method_name} method in the following class {class_name}\n\n"
        prompt += class_text_desc + "\n\n"
        prompt += "Only provide the method implementation without any explanation."
        # print("prompt:")
        # print(prompt)
        if USE_DS_API:
            generated_text = ds_api_generate(prompt, client)
        else:
            generated_text = model_generate(prompt, model, tokenizer)
        code = extract_python_code(generated_text)
        pure_code = extract_pure_code(code)
        # print("code:")
        # print(code)
        pure_code_lines = pure_code.split("\n")
        class_text += "\n\n" + method_desc
        for code_line in pure_code_lines:
            class_text += "\n" + code_line

    return class_text


def main():
    for problem in data:
        task_id = problem["task_id"]
        id = int(task_id.split("_")[-1])
        print(f"generating {task_id}")
        final_class_text = process_single_sample(problem)
        # print("final class:")
        # print(final_class_text)
        with open(os.path.join(output_dir, f"{task_id}.py"), "w") as f:
            f.write(final_class_text)


if __name__ == "__main__":
    main()