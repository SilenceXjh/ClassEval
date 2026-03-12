import ast
import sys
import json


def extract_imports(code: str):
    """
    解析 import 语句
    """
    try:
        tree = ast.parse(code)
    except Exception as e:
        print(e)
        print(code)
        return []
    modules = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                modules.add(n.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.add(node.module.split('.')[0])

    std_lib = sys.stdlib_module_names
    modules = list(modules)
    third_party = [m for m in modules if m not in std_lib]

    return third_party


def main():
    code_file = "/data0/xjh/ClassEval/custom_generation/holistic.json"
    third_party_set = set()
    with open(code_file, "r") as f:
        data = json.load(f)
        for sample in data:
            code = sample["predict"][0]
            third_party = extract_imports(code)
            third_party_set.update(third_party)

    print(list(third_party_set))

main()