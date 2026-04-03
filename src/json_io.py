
import json



def generate_json_output(output: list[dict[str, str]], output_path: str) -> None:
    data = [{
            "prompt": dic["prompt"],
            "name":  dic["func_name"],
            "parameters": dic["parameters"]
        } for dic in output]
        
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

def get_prompt(input_path: str) -> list[str]:
    with open(input_path, "r") as f:
        data = json.loads(f.read())
    prompts = [item["prompt"] for item in data]
    return prompts

def get_func_def(input_path: str) -> list[str]:
    with open(input_path, "r") as f:
        functions = json.loads(f.read())
    return functions
