import json
import os

OUTPUT_DIR = "test_dir"
os.makedirs(OUTPUT_DIR)

model_output_file_path = "../generation/model_output.json"

with open(model_output_file_path, "r", encoding = 'utf-8') as f:
    file_content = json.load(f)
    for cont in file_content:
        pred = cont["predict"][0]
        test_cases = cont["test"]
        task_id = cont["task_id"]
        with open(os.path.join(OUTPUT_DIR, f"{task_id}.py"), "w") as test_f:
            test_f.write(pred)
            test_f.write("\n\n")
            test_f.write(test_cases)
            test_f.write("\n\nif __name__ == '__main__':\n    import unittest\n    unittest.main()\n")