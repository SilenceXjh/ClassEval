import os
import json
import subprocess
import tempfile
from pathlib import Path
import ast

DOCKER_IMAGE = "my-python-runtime:1.0"

script_suffix = """if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)

    total = result.testsRun
    fails = len(result.failures)
    errors = len(result.errors)
    passed = total - fails - errors
    
    if passed == total:
        print("ALL TESTS PASSED")
    
    statistics = {
        "total": total,
        "fails": fails,
        "errors": errors,
        "passed": passed
    }
    print(statistics)
"""


def construct_file_content(code_str, test_str):
    return code_str + "\n\n" + test_str + "\n\n" + script_suffix

def run_single_sample(code_str, test_str):
    with tempfile.TemporaryDirectory(dir="/data0/xjh/tmp") as tmpdir:
        tmpdir = Path(tmpdir)

        (tmpdir / "solution.py").write_text(construct_file_content(code_str, test_str))
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
                capture_output=True,
                text=True,
                timeout=10
            )

        except Exception as e:
            print(e)
            return False, None

        finally:
            # 4. 清理：无论成功失败，都删除容器
            subprocess.run(["docker", "rm", "-f", "temp_worker"], capture_output=True)

    if result.returncode != 0:
        print("执行失败:", result.stderr)
        print("stdout:", result.stdout)
        return False, None
    
    inner_output = result.stdout
    start_idx = inner_output.rfind('{')
    end_idx = inner_output.rfind('}')
    dict_str = inner_output[start_idx : end_idx + 1]
    data_dict = ast.literal_eval(dict_str)

    if "ALL TESTS PASSED" in result.stdout:
        print("测试通过")
        return True, data_dict
    
    print("测试失败:", result.stdout)
    return False, data_dict


def evaluate(code_path, data_path):
    with open(data_path, "r") as f:
        data = json.load(f)

    total = 0
    right = 0

    testcase_total = 0
    testcase_passed = 0

    for sample in data:
        task_id = sample["task_id"]
        with open(os.path.join(code_path, f"{task_id}.py"), "r") as f:
            code = f.read()
        
        test_code = sample["test"]

        success, data_dict = run_single_sample(code, test_code)
        if success:
            right += 1

        if data_dict:
            testcase_total += data_dict["total"]
            testcase_passed += data_dict["passed"]

    print("total:", total)
    print("right:", right)

    print("testcase total:", testcase_total)
    print("testcase passed:", testcase_passed)

def main():
    code_path = "/data0/xjh/ClassEval/custom_generation/qwen7B_holistic"
    data_path = "/data0/xjh/ClassEval/custom_generation/data/ClassEval_data.json"
    evaluate(code_path, data_path)

if __name__ == "__main__":
    main()