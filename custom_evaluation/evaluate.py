import os
import json
import subprocess
import tempfile
from pathlib import Path

DOCKER_IMAGE = "python:3.10-slim"

with open("install_dependencies.py") as f:
    install_dependencies_code = f.read()

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

def run_single_sample(code_str, test_str) -> bool:
    with tempfile.TemporaryDirectory(dir="/data0/xjh/tmp") as tmpdir:
        tmpdir = Path(tmpdir)

        (tmpdir / "solution.py").write_text(construct_file_content(code_str, test_str))
        (tmpdir / "install_dependencies.py").write_text(install_dependencies_code)

        create_cmd = [
            "docker", "create",
            "--name", "temp_worker",  # 指定容器名称
            "--workdir", "/app",
            "-e", "http_proxy=http://210.28.134.91:8888",
            "-e", "https_proxy=http://210.28.134.91:8888",
            DOCKER_IMAGE,
            "sh", "-c", "python install_dependencies.py && python solution.py"
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
                timeout=20
            )

        finally:
            # 4. 清理：无论成功失败，都删除容器
            subprocess.run(["docker", "rm", "-f", "temp_worker"], capture_output=True)

        # result = subprocess.run(
        #     [
        #         "docker", "run",
        #         "--rm",
        #         "-v", f"{tmpdir}:/app",
        #         '-w', '/app',
        #         "--network", "host",
        #         "--dns", "8.8.8.8",
        #         "--dns", "114.114.114.114",
        #         "-e", "http_proxy=http://210.28.134.91:8888",
        #         "-e", "https_proxy=http://210.28.134.91:8888",
        #         DOCKER_IMAGE,
        #         "sh", "-c", "python install_dependencies.py && python solution.py"
        #     ],
        #     capture_output=True,
        #     text=True,
        #     timeout=20
        # )

    if result.returncode != 0:
        print("执行失败:", result.stderr)
        print("stdout:", result.stdout)
        return False

    if "ALL TESTS PASSED" in result.stdout:
        print("测试通过")
        return True
    
    print("测试失败:", result.stdout)
    return False


def evaluate(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    total = 0
    right = 0

    for sample in data:
        total += 1

        code = sample["predict"][0]
        test_code = sample["test"]

        res = run_single_sample(code, test_code)
        if res:
            right += 1

    print("total:", total)
    print("right:", right)


def main():
    output_file_path = "/data0/xjh/ClassEval/generation/model_output.json"
    evaluate(output_file_path)

if __name__ == "__main__":
    main()