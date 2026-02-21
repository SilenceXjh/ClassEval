import re
from enum import Enum

class ModelName(Enum):
    DEEPSEEK_API = 0
    QWEN_CODER = 1
    QWEN_CODER_INST = 2

class GenerationStrategy(Enum):
    Holistic = 0
    Incremental = 1
    Compositional = 2

class InferenceUtil:

    @staticmethod
    def generate_prompt(instruction, model_name):
        if model_name == ModelName.DEEPSEEK_API.value or model_name == ModelName.QWEN_CODER_INST.value:
            return instruction

        else:
            raise NotImplementedError()

    @staticmethod
    def get_leading_spaces(string):
        return len(string) - len(string.lstrip())

    @staticmethod
    def del_segment_notation(code):
        pattern = r'(""".*?""")'
        result = re.sub(pattern, '', code, flags = re.DOTALL)
        return result

    @staticmethod
    def get_method_signature(code, method_name):
        method_def_prefix = "def " + method_name + '('
        code_segment = code.split('):')
        for segment in code_segment:
            if method_def_prefix in segment:
                return "    " + segment + "):"
        return ""

    @staticmethod
    def add_desc_to_init(desc, class_init):
        class_init_list = class_init.split('\n')
        class_init_list[0] += " \n" + desc
        class_init = '\n'.join(class_init_list)
        return class_init

    @staticmethod
    def extract_method_code(code, method_name):
        # extract code of method {method_name} from {code}
        output_split_identifier_list = ["### Response:", "@@ Response:", "[/INST]"]
        for identifier in output_split_identifier_list:
            if identifier in code:
                code = code.split(identifier)[1]
                break

        pattern_list = [r"```python(.*?)```", r"\[PYTHON\](.*?)\[/PYTHON\]"]
        for pattern in pattern_list:
            code_part = re.findall(pattern, code, re.S)
            if code_part:
                code = code_part[0]
                break

        code_list = code.split('\n')

        method_code_list = []
        method_def_prefix = "def " + method_name + '('
        skip_line_list = ["```", '\r']
        # extract generated method code corresponding method_name, the strategy is to find the line
        # has "def methodname(...)" and following lines have more leading spaces than the first "def" line
        for i, line in enumerate(code_list):
            if method_def_prefix in line:
                method_code_list = code_list[i:]
                break

        if len(method_code_list) == 0:
            return ""

        for i, line in enumerate(method_code_list):
            if line in skip_line_list:
                method_code_list[i] = ""
        
        if InferenceUtil.get_leading_spaces(method_code_list[1]) - InferenceUtil.get_leading_spaces(method_code_list[0]) > 4:
            method_code_list[0] = " " * 4 + method_code_list[0]

        first_line_leading_space = InferenceUtil.get_leading_spaces(method_code_list[0])
        for i, line in enumerate(method_code_list[1:]):
            if InferenceUtil.get_leading_spaces(line) <= first_line_leading_space and len(line) > 0:
                method_code_list = method_code_list[:i + 1]
                break

        for i, line in enumerate(method_code_list):
            method_code_list[i] = ' ' * (4 - first_line_leading_space) + line
        
        if 'self' not in method_code_list[0] and 'cls' not in method_code_list[0]:
            method_code_list.insert(0, ' ' * 4 + "@staticmethod")

        line_notation_mark = 0
        for line in method_code_list:
            if line == " " * 8 + "\"\"\"" or line == " " * 4 + "\"\"\"":
                line_notation_mark = line_notation_mark + 1
        if line_notation_mark % 2 == 1:
            method_code_list.append(" " * 8 + "\"\"\"")
            method_code_list.append(" " * 8 + "pass")

        method_code = '\n'.join(method_code_list)
        method_code = method_code.rstrip() + '\n'
        return method_code
    
    @staticmethod
    def extract_python_code(generated_text: str) -> str:
        """从生成的文本中提取 py 代码"""
        if "```python" in generated_text:
            code = generated_text.split("```python")[1].split("```")[0].strip()
        elif "```" in generated_text:
            code = generated_text.split("```")[1].split("```")[0].strip()
        else:
            code = generated_text.strip()

        return code