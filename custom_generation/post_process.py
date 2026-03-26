import os
import re
import shutil
from pathlib import Path

def remove_test_code(file_path: str, backup: bool = True) -> None:
    """
    删除Python文件中的unittest测试类和__main__代码块
    
    Args:
        file_path: 目标Python文件路径
        backup: 是否创建备份文件（默认创建，备份文件后缀为.bak）
    """
    # 验证文件是否存在
    file = Path(file_path)
    if not file.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 创建备份
    if backup:
        backup_path = f"{file_path}.bak"
        shutil.copy2(file_path, backup_path)
        print(f"已创建备份文件: {backup_path}")
    
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 标记需要保留的行
    keep_lines = [True] * len(lines)
    
    # 1. 删除继承自unittest.TestCase的测试类
    in_test_class = False
    class_indent_level = 0
    
    for i, line in enumerate(lines):
        # 匹配测试类定义行 (例如: class TestXXX(unittest.TestCase):)
        class_match = re.match(r'^(\s*)class\s+\w+\(.*unittest\.TestCase.*\):', line)
        if class_match:
            in_test_class = True
            class_indent_level = len(class_match.group(1))
            keep_lines[i] = False
            continue
        
        # 如果在测试类内部
        if in_test_class:
            keep_lines[i] = False
            
            # 检查是否退出测试类（缩进级别回到类定义级别或更低）
            current_indent = len(re.match(r'^\s*', line).group(0))
            # 空行或缩进级别小于类定义级别，表示类结束
            if current_indent <= class_indent_level and line.strip() != '':
                in_test_class = False
    
    # 2. 删除if __name__ == '__main__': 代码块
    in_main_block = False
    main_indent_level = 0
    
    for i, line in enumerate(lines):
        # 匹配__main__代码块开始
        main_match = re.match(r'^(\s*)if __name__ == [\'"]__main__[\'"]:', line)
        if main_match:
            in_main_block = True
            main_indent_level = len(main_match.group(1))
            keep_lines[i] = False
            continue
        
        # 如果在__main__代码块内部
        if in_main_block:
            keep_lines[i] = False
            
            # 检查是否退出__main__代码块
            current_indent = len(re.match(r'^\s*', line).group(0))
            # 缩进级别回到__main__定义级别或更低，且不是空行，表示代码块结束
            if current_indent <= main_indent_level and line.strip() != '':
                in_main_block = False
    
    # 过滤并保留需要的行
    filtered_lines = [line for i, line in enumerate(lines) if keep_lines[i]]
    
    # 去除多余的空行（可选优化）
    final_lines = []
    prev_empty = False
    for line in filtered_lines:
        stripped = line.strip()
        if not stripped:
            if not prev_empty:
                final_lines.append(line)
                prev_empty = True
        else:
            final_lines.append(line)
            prev_empty = False
    
    # 将处理后的内容写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(final_lines)
    
    print(f"处理完成！文件: {file_path}")
    print(f"删除了 {len(lines) - len(final_lines)} 行内容")

# 使用示例
if __name__ == '__main__':
    # 替换为你要处理的文件路径
    code_dir = "/data0/xjh/ClassEval/custom_generation/qwen1.5b_tdd"
    for file_name in os.listdir(code_dir):
        target_file = os.path.join(code_dir, file_name)
        try:
            remove_test_code(target_file)
            print("操作成功完成！")
        except Exception as e:
            print(f"处理过程中出错: {e}")