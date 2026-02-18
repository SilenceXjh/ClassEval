import sys
from install_dependencies import extract_imports

code_path = "test_dir/ClassEval_1.py"

with open(code_path, "r") as f:
    code = f.read()

import_modules = extract_imports(code)
print("import modules:", import_modules)

std_lib = sys.stdlib_module_names
third_party = [m for m in import_modules if m not in std_lib]
print("third party:", third_party)