import ast
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

DOCKER_IMAGE = "my-python-runtime:1.0"  # 替换为你的镜像名

script_suffix = """\

if __name__ == '__main__':
    import sys, json as _json

    _suite = unittest.TestLoader().loadTestsFromModule(__import__(__name__))

    # 收集所有测试类名
    _class_names = []
    for _group in _suite:
        for _test in _group:
            _cname = _test.__class__.__name__
            if _cname not in _class_names:
                _class_names.append(_cname)

    _class_results = {}
    _mod = __import__(__name__)

    for _cname in _class_names:
        _cls_suite = unittest.TestLoader().loadTestsFromTestCase(getattr(_mod, _cname))
        _buf = __import__('io').StringIO()
        _runner = unittest.TextTestRunner(verbosity=0, stream=_buf)
        _r = _runner.run(_cls_suite)

        _total  = _r.testsRun
        _fails  = len(_r.failures)
        _errors = len(_r.errors)
        _passed = _total - _fails - _errors

        _class_results[_cname] = {
            "total":        _total,
            "passed":       _passed,
            "failed":       _fails,
            "errors":       _errors,
            "class_passed": (_passed == _total and _total > 0),
        }

    _method_total  = sum(v["total"]  for v in _class_results.values())
    _method_passed = sum(v["passed"] for v in _class_results.values())
    _method_failed = sum(v["failed"] for v in _class_results.values())
    _method_errors = sum(v["errors"] for v in _class_results.values())

    _class_total  = len(_class_results)
    _class_passed = sum(1 for v in _class_results.values() if v["class_passed"])

    _statistics = {
        "status": "ok",
        "method_level": {
            "total":  _method_total,
            "passed": _method_passed,
            "failed": _method_failed,
            "errors": _method_errors,
        },
        "class_level": {
            "total":  _class_total,
            "passed": _class_passed,
            "failed": _class_total - _class_passed,
        },
        "class_details": _class_results,
    }
    print(_json.dumps(_statistics))
"""


def _count_test_classes(source: str) -> int:
    """从源码中统计 unittest.TestCase 子类的数量，用于语法错误/超时时兜底。"""
    return len(re.findall(r"^class\s+\w+\s*\(.*?TestCase.*?\)\s*:", source, re.MULTILINE))


def _count_test_methods(source: str) -> int:
    """从源码中统计 test_ 开头的测试方法数量。"""
    return len(re.findall(r"^\s+def\s+(test_\w+)\s*\(", source, re.MULTILINE))


def construct_file_content(code_str: str, test_str: str) -> str:
    return code_str + "\n\n" + test_str + "\n\n" + script_suffix


def _make_error_result(status: str, n_classes: int, n_methods: int, extra: dict | None = None) -> dict:
    result = {
        "status": status,
        "method_level": {"total": n_methods, "passed": 0, "failed": 0, "errors": n_methods},
        "class_level":  {"total": n_classes, "passed": 0, "failed": n_classes},
        "class_details": {},
    }
    if extra:
        result.update(extra)
    return result


def run_single_sample(code_str: str, test_str: str):
    """
    返回 (all_passed: bool, statistics: dict)
    statistics 结构:
    {
        "status":       "ok" | "syntax_error" | "timeout" | "runtime_error",
        "method_level": {"total", "passed", "failed", "errors"},
        "class_level":  {"total", "passed", "failed"},
        "class_details": { ClassName: {"total","passed","failed","errors","class_passed"}, ... }
    }
    """
    file_content = construct_file_content(code_str, test_str)
    n_classes = _count_test_classes(test_str)
    n_methods = _count_test_methods(test_str)

    # ── 1. 语法检查（在宿主机上做，避免无谓地拉起容器）──────────────────────────
    try:
        compile(file_content, "solution.py", "exec")
    except SyntaxError as e:
        stats = _make_error_result("syntax_error", n_classes, n_methods, {"error": str(e)})
        return False, stats

    # ── 2. 写文件、拉起容器 ───────────────────────────────────────────────────
    container_name = f"temp_worker_{id(file_content)}"   # 简单避免并发冲突
    result_proc = None

    with tempfile.TemporaryDirectory(dir="/data0/xjh/tmp") as tmpdir:
        (Path(tmpdir) / "solution.py").write_text(file_content, encoding="utf-8")

        create_cmd = [
            "docker", "create",
            "--name", container_name,
            "--workdir", "/app",
            DOCKER_IMAGE,
            "sh", "-c", "python solution.py",
        ]

        try:
            subprocess.run(create_cmd, check=True, capture_output=True)
            subprocess.run(
                ["docker", "cp", f"{tmpdir}/.", f"{container_name}:/app"],
                check=True, capture_output=True, text=True,
            )
            result_proc = subprocess.run(
                ["docker", "start", "-a", container_name],
                capture_output=True, text=True,
                timeout=30,
            )

        except subprocess.TimeoutExpired:
            stats = _make_error_result("timeout", n_classes, n_methods)
            return False, stats

        except Exception as e:
            stats = _make_error_result("runtime_error", n_classes, n_methods, {"error": str(e)})
            return False, stats

        finally:
            subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)

    # ── 3. 解析输出 ───────────────────────────────────────────────────────────
    if result_proc is None:
        return False, _make_error_result("runtime_error", n_classes, n_methods)

    if result_proc.returncode != 0:
        # 运行时错误（导入失败、NameError 等）
        stats = _make_error_result(
            "runtime_error", n_classes, n_methods,
            {"stderr": result_proc.stderr, "stdout": result_proc.stdout},
        )
        return False, stats

    # 从 stdout 最后一行合法 JSON 中提取统计
    for line in reversed(result_proc.stdout.strip().splitlines()):
        try:
            stats = json.loads(line)
            all_passed = (
                stats.get("class_level", {}).get("passed", 0)
                == stats.get("class_level", {}).get("total", -1)
                and stats["class_level"]["total"] > 0
            )
            return all_passed, stats
        except (json.JSONDecodeError, KeyError):
            continue

    # stdout 里没有 JSON（理论上不应出现，防御性兜底）
    stats = _make_error_result(
        "runtime_error", n_classes, n_methods, 
        {"stderr": result_proc.stderr, "stdout": result_proc.stdout},
    )
    return False, stats


def evaluate(code_path, data_path):
    with open(data_path, "r") as f:
        data = json.load(f)

    total = 0
    right = 0
    test_class_total = 0
    test_class_passed = 0
    method_total = 0
    method_passed = 0

    for sample in data:
        task_id = sample["task_id"]
        with open(os.path.join(code_path, f"{task_id}.py"), "r") as f:
            code = f.read()

        test_code = sample["test"]

        total += 1
        success, stats = run_single_sample(code, test_code)

        if stats is not None:
            test_class_total  += stats["class_level"]["total"]
            test_class_passed += stats["class_level"]["passed"]
            method_total  += stats["method_level"]["total"]
            method_passed += stats["method_level"]["passed"]

        if success:
            right += 1

        # 进度打印
        status_tag = stats.get("status", "unknown") if stats else "unknown"
        print(f"[{total:>3}/{len(data)}] {task_id}  success={success}  status={status_tag}  "
              f"class={stats['class_level']['passed']}/{stats['class_level']['total']}  "
              f"method={stats['method_level']['passed']}/{stats['method_level']['total']}"
              if stats else
              f"[{total:>3}/{len(data)}] {task_id}  success={success}  status=unknown")

    # 汇总
    print("\n" + "=" * 60)
    print(f"题目总数        : {total}")
    print(f"题目通过数      : {right}  ({100 * right / total:.1f}%)" if total else "题目通过数: 0")
    print(f"测试类总数      : {test_class_total}")
    print(f"测试类通过数    : {test_class_passed}  ({100 * test_class_passed / test_class_total:.1f}%)" if test_class_total else "测试类通过数: 0")
    print(f"测试方法总数    : {method_total}")
    print(f"测试方法通过数  : {method_passed}  ({100 * method_passed / method_total:.1f}%)" if method_total else "测试方法通过数: 0")
    print("=" * 60)
        

    

def main():
    code_path = "/data0/xjh/ClassEval/custom_generation/ds_tdd"
    data_path = "/data0/xjh/ClassEval/data/ClassEval_data.json"
    evaluate(code_path, data_path)

if __name__ == "__main__":
    main()