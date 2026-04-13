from llm_sdk import Small_LLM_Model
from json_io import generate_json_output, get_prompt, get_func_def
from function_call import FunctionCall
from argparse import ArgumentParser, Namespace


def get_input_path()-> Namespace:
    parser = ArgumentParser(exit_on_error=False)
    parser.add_argument("--input_function",
                        help="the path to func definitions",
                        default="data/input/functions_definition.json",
                        required=False)
    parser.add_argument("--input_prompt",
                        help="the path to prompt list",
                        default="data/input/function_calling_tests.json",
                        required=False)
    parser.add_argument("--output_file",
                        help="where to write the output",
                        default="data/output/output.txt",
                        required=False)
    return parser.parse_args()


def main()-> None:
    try:
        path = get_input_path()
        function_defs= get_func_def(path.input_function)
        prompts_list = get_prompt(path.input_prompt)
        llm = Small_LLM_Model(model_name="model/qwen")
    except Exception as error:
        print(error, "occured while initializing program")
        return
    future_json = []
    for prompt in prompts_list:
        act_call = FunctionCall(llm, function_defs, prompt)
        print(f"resolving prompt : {prompt}")
        act_call.find_fn_name()
        act_call.get_param()
        future_json.append(act_call.to_dict())
    print("result :\n" +"\n\n".join("".join('\n'.join([f'[{k} : {v}]' for k, v in dic.items()])) for dic in future_json))
    generate_json_output(future_json, path.output_file)

if __name__ == "__main__":
    main()
