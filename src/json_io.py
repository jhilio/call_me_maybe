
import json



def generate_json_output(output: list[dict[str, str]], output_path: str="data/output/out.txt") -> None:
    data = [{
            "prompt": dic["prompt"],
            "name":  dic["func_name"],
            "parameters": dic["parameters"]
        } for dic in output]
        
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

def get_prompt(input_path: str="data/input/function_calling_tests.json") -> list[str]:
    with open(input_path, "r") as f:
        data = json.loads(f.read())
    prompts = [item["prompt"] for item in data]
    return prompts