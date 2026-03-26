import ast
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import model_generate, extract_python_code

data_path = "/data0/xjh/ClassEval/data/ClassEval_data.json"
model_path = "/data1/model/qwen/Qwen/Qwen2.5-Coder-7B-Instruct/"
output_dir = "/data0/xjh/ClassEval/custom_generation/qwen7b_incremental_test_first"

os.makedirs(output_dir, exist_ok=True)

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


def trim_test_class(source: str) -> str:
    """
    保留 setUp/tearDown 方法和第一个 test 方法，删除其余 test 方法和 import 语句。
    """
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    lines_to_remove = set()

    for node in ast.walk(tree):
        # 删除 import 语句
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for i in range(node.lineno - 1, node.end_lineno):
                lines_to_remove.add(i)

        # 处理类中的 test 方法
        if not isinstance(node, ast.ClassDef):
            continue

        first_test_seen = False
        for item in node.body:
            if not isinstance(item, ast.FunctionDef):
                continue
            if item.name.startswith('test'):
                if not first_test_seen:
                    first_test_seen = True
                else:
                    for i in range(item.lineno - 1, item.end_lineno):
                        lines_to_remove.add(i)

    result_lines = [
        line for i, line in enumerate(lines)
        if i not in lines_to_remove
    ]

    return ''.join(result_lines)


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
        # print("class_text_desc:")
        # print(class_text_desc)

        # generate some testcases for the method
        method_tests = method_info["test_code"]
        trimed_method_tests = trim_test_class(method_tests)
        gen_test_prompt = f"""Please generate some testcases for method {method_name} in the following class {class_name}
```
{class_text_desc}
```

You can generate testcase like this:
```
{trimed_method_tests}
```
Generated 3-5 testcases in a 'unittest.TestCase' class. Only output the generated testcases without any explanation."""
        print("------gen test prompt------")
        print(gen_test_prompt)
        generated_text = model_generate(gen_test_prompt, model, tokenizer)
        gen_tests = extract_python_code(generated_text)
        print("------gen tests------")
        print(gen_tests)

        prompt = f"""Please complete {method_name} method in the following class {class_name}
```
{class_text_desc}
```

Your implementation should pass the following tests:
```
{gen_tests}
```
Only provide the method implementation without any explanation.
"""
        print("------prompt------")
        print(prompt)

        generated_text = model_generate(prompt, model, tokenizer)
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
        with open(os.path.join(output_dir, f"{task_id}.py"), "w") as f:
            f.write(final_class_text)


if __name__ == "__main__":
    main()