import ast
import json
import os
from pathlib import Path
import subprocess
import tempfile
from openai import OpenAI
from utils import extract_python_code, ds_api_generate

data_path = "/data0/xjh/ClassEval/data/ClassEval_data.json"
output_dir = "/data0/xjh/ClassEval/custom_generation/ds_tdd"

USE_DS_API = True

os.makedirs(output_dir, exist_ok=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

with open(data_path, "r") as f:
    data = json.load(f)

client = OpenAI(api_key=os.environ.get('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")


DOCKER_IMAGE = "my-python-runtime:1.0"

script_suffix = """if __name__ == '__main__':
    import inspect
    suite = unittest.TestLoader().loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)

    total = result.testsRun
    fails = len(result.failures)
    errors = len(result.errors)
    passed = total - fails - errors
    
    if passed == total:
        print("ALL TESTS PASSED")
    else:
        failure_datails = []

        for test, tb in result.failures + result.errors:

            method = getattr(test, test._testMethodName)

            try:
                method_source = inspect.getsource(method)
            except:
                method_source = None

            failure_datails.append({
                "failed_test_method": method_source,
                "traceback": tb,
            })

        print("unit tests failure details:")
        print(failure_datails)
    
"""


def construct_file_content(code_str, test_str):
    return code_str + "\n\n" + "import unittest\n\n" + test_str + "\n\n" + script_suffix

def run_single_sample(code_str, test_str):
    with tempfile.TemporaryDirectory(dir="/data0/xjh/tmp") as tmpdir:
        tmpdir = Path(tmpdir)

        file_content = construct_file_content(code_str, test_str)
        # with open("/data0/xjh/bigcodebench-experiment/playground/a.py", "w") as f:
        #     f.write(file_content)
        (tmpdir / "solution.py").write_text(file_content)
        # (tmpdir / "install_dependencies.py").write_text(install_dependencies_code)

        create_cmd = [
            "docker", "create",
            "--name", "temp_worker",  # 指定容器名称
            "--workdir", "/app",
            DOCKER_IMAGE,
            "sh", "-c", "python solution.py"
        ]

        try:
            # 创建容器并获取 ID
            subprocess.run(create_cmd, check=True, capture_output=True)

            # 2. 拷贝文件 (docker cp)
            # 直接将整个临时目录下的内容拷贝到容器的 /app 目录
            # 注意：src 路径最后加个 / 会拷贝目录下的内容，而不是目录本身
            subprocess.run(["docker", "cp", f"{tmpdir}/.", "temp_worker:/app"], 
                           check=True,
                           capture_output=True,
                           text=True)

            # 3. 启动并等待结果 (docker start)
            # -a (attach) 会让 subprocess 等待容器运行结束并获取输出
            result = subprocess.run(
                ["docker", "start", "-a", "temp_worker"],
                text=True,
                timeout=120,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )

        except Exception as e:
            print(e)
            return False, "execution timeout"

        finally:
            # 4. 清理：无论成功失败，都删除容器
            subprocess.run(["docker", "rm", "-f", "temp_worker"], capture_output=True)

    if result.returncode != 0:
        # print("执行失败:", result.stdout)

        return False, result.stdout
    

    if "ALL TESTS PASSED" in result.stdout:
        # print("测试通过")
        return True, None
    
    # print("测试失败:", result.stdout)
    if "unit tests failure details:" in result.stdout:
        try:
            error_msg = result.stdout.split("unit tests failure details:")[-1].strip()
            data_obj = ast.literal_eval(error_msg.strip())
            formatted_json = json.dumps(data_obj, indent=2, ensure_ascii=False)
            feedback_msg = "Some unit tests fail:\n" + formatted_json
        except Exception:
            feedback_msg = result.stdout
    else:
        feedback_msg = result.stdout
    return False, feedback_msg


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
    # assert def_line_no != -1
    if lines[def_line_no].startswith("def"):
        need_indent = True
    else:
        need_indent = False
    
    # 标记是否在文档字符串中
    in_docstring = False
    docstring_start = '"""'
    result_lines = []
    
    for line in lines[def_line_no+1:]:
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
        result_lines.append(line)
        
        
    
    # 去除结果中首尾的空白行
    while result_lines and result_lines[0].strip() == '':
        result_lines.pop(0)
    while result_lines and result_lines[-1].strip() == '':
        result_lines.pop()
    
    # 拼接成最终的代码块
    return '\n'.join(result_lines)


def construct_repair_prompt(code: str, test_code: str, feedback: str):
    prompt = f"""Please fix a python class with some methods, based on the current code, test code, and test feedback.
        
### Current code:
```python
{code}
```

### Test code:
```
{test_code}
```

### Test feedBack: 
{feedback}

Don't change the class and methods signature. Only output the fixed class implementation."""
    return prompt


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
        prompt += "Only provide the complete method implementation. The method implementation you output must contain method head. " + \
        "Don't output any other content."
        # print("prompt:")
        # print(prompt)

        generated_text = ds_api_generate(prompt, client)
        code = extract_python_code(generated_text)
        pure_code = extract_pure_code(code)
        print("code:")
        print(code)
        pure_code_lines = pure_code.split("\n")
        class_text += "\n\n" + method_desc
        for code_line in pure_code_lines:
            class_text += "\n" + code_line
        print("class text:")
        print(class_text)
        
        method_tests = method_info["test_code"]
        success, feedback = run_single_sample(class_text, method_tests)
        if not success:
            for i in range(3):
                repair_prompt = construct_repair_prompt(class_text, method_tests, feedback)
                generated_text = ds_api_generate(repair_prompt, client)
                class_text = extract_python_code(generated_text)
                success, feedback = run_single_sample(class_text, method_tests)
                if success:
                    break


    return class_text


def main():
    for problem in data[3:]:
        task_id = problem["task_id"]
        final_class_text = process_single_sample(problem)
        # print("final class:")
        # print(final_class_text)
        with open(os.path.join(output_dir, f"{task_id}.py"), "w") as f:
            f.write(final_class_text)


if __name__ == "__main__":
    main()