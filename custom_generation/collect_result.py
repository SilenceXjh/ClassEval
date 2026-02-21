import json


data_path = "/data0/xjh/ClassEval/data/ClassEval_data.json"

with open(data_path, "r") as f:
    data = json.load(f)

result = []

for problem in data:
    tests = problem["test"]
    task_id = problem["task_id"]
    with open(f"{task_id}.py", "r") as f:
        solution = f.read()
    prediction = [solution]
    result.append({
        "task_id": task_id,
        "predict": prediction,
        "test": tests,
    })

with open("holistic.json", "w") as f:
    json.dump(result, f, indent=2)