import sys
import subprocess
import ast

CODE_FILE = "solution.py"

IMPORT_TO_PIP = {
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "bs4": "beautifulsoup4",
    "jwt": "PyJWT",
    "PIL": "Pillow",
    "yaml": "PyYAML",
    "dateutil": "python-dateutil",
    "Crypto": "pycryptodome",
}


def extract_imports(code):
    """
    解析 import 语句
    """
    tree = ast.parse(code)
    modules = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                modules.add(n.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.add(node.module.split('.')[0])

    return list(modules)


def install_dependencies(modules):
    """
    安装第三方依赖
    """
    std_lib = sys.stdlib_module_names

    third_party = [m for m in modules if m not in std_lib]

    if not third_party:
        return
    
    corrected_modules = []
    for m in third_party:
        if m in IMPORT_TO_PIP:
            corrected_modules.append(IMPORT_TO_PIP[m])
        else:
            corrected_modules.append(m)

    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-cache-dir"] + corrected_modules,
        check=True,  # 如果安装失败直接抛出异常
        capture_output=True,
        text=True
    )
    print("pip install cmd result:", result.returncode, result.stdout, result.stderr)
    print("pip install:", corrected_modules)


def main():
    try:
        with open(CODE_FILE) as f:
            code = f.read()

        modules = extract_imports(code)
        install_dependencies(modules)

    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
