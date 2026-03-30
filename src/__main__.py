
from llm_sdk import Small_LLM_Model
from json_io import generate_json_output, get_prompt
from function_call import FunctionCall
import json






        

def main():
    llm = Small_LLM_Model(model_name="model/qwen")
    # List of allowed strings
    with open("data/input/functions_definition.json") as file:
        content = file.read()
    data = json.loads(content)

    # fn_names.append("None")
    future_json = []
    prompts_list = get_prompt()
    for prompt in prompts_list:
        act_call = FunctionCall(llm, data, prompt)
        print(f"resolving prompt : {prompt}")
        act_call.find_fn_name()
        act_call.get_param()
        future_json.append(act_call.to_dict())
    print("result :\n" +''.join(f"{'\n'.join(f'[{k} : {v}]' for k, v in dic.items())}\n\n" for dic in future_json))
    generate_json_output(future_json)

if __name__ == "__main__":
    main()
