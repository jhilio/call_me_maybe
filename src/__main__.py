from llm_sdk import Small_LLM_Model
from llm_interaction import MyLLM
from json_io import generate_json_output, get_prompt, get_func_def
from function_call import FunctionCall
from argparse import ArgumentParser, Namespace
from utils import monitor_time

def get_input_path() -> Namespace:
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


@monitor_time
def main()-> None:
    # try:
    path = get_input_path()
    function_defs = get_func_def(path.input_function)
    prompts_list = get_prompt(path.input_prompt)
    llm = MyLLM(Small_LLM_Model())
    # except Exception as error:
    #     print(error, "occured while initializing program")
    #     return
    try:
        future_json = []
        for prompt in prompts_list:
            act_call = FunctionCall(llm, function_defs, prompt)
            print(f"resolving prompt : {prompt}")
            act_call.find_fn_name()
            act_call.get_param()
            future_json.append(act_call.to_dict())
        print("result :\n" +"\n\n".join("".join('\n'.join([f'[{k} : {v}]' for k, v in dic.items()])) for dic in future_json))
        generate_json_output(future_json, path.output_file)
    except IOError as error:
        print(error, "occured while running program")

# def main()-> None:
#     llm = Small_LLM_Model()
#     my_version = MyLLM(llm)
#     for arg in sys.argv:
#         test1 = llm.encode(arg)
#         test2 = my_version.encode(arg)
#         print(test1, "\n" + str(test2), "\n" + "\n" + my_version.decode(test2.tolist()[0]))


if __name__ == "__main__":
    main()
    # stats = monitor_time(show=True)
    # for func, s in sorted(stats.items(), key=lambda m: m[1]["total_time"]):
    #     if s["call_count"]:
    #         avg = s["total_time"] / s["call_count"]
    #         print(f"{func.__name__}, took a total of :\n{s['total_time']:.3f}sec, for {s['call_count']} call, with an average of {avg:.3f}sec\n")