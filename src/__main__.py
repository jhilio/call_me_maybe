from llm_sdk import Small_LLM_Model
from llm_interaction import MyLLM
from json_io import generate_json_output, get_prompt, get_func_def
from function_call import FunctionCall
from argparse import ArgumentParser, Namespace
from utils import monitor_time, format_dict


def get_input_path() -> Namespace:
    """uses argparse to read program argument

    Returns:
        Namespace:
            filled with all path to the files specified at program launch
    """
    parser = ArgumentParser(exit_on_error=False)
    parser.add_argument("--functions_definition",
                        help="the path to func definitions",
                        default="data/input/functions_definition.json",
                        required=False)
    parser.add_argument("--input",
                        help="the path to prompt list",
                        default="data/input/function_calling_tests.json",
                        required=False)
    parser.add_argument("--output",
                        help="where to write the output",
                        default="data/output/function_calling_results.json",
                        required=False)
    parser.add_argument("--show_time",
                        help="weither to show time at end of programe",
                        default="false",
                        required=False)
    return parser.parse_args()


@monitor_time
def main() -> bool:
    """starting point of programe

    Returns:
        weither to show time measured
    """
    try:
        path = get_input_path()
        function_defs = get_func_def(path.functions_definition)
        prompts_list = get_prompt(path.input)
        llm = MyLLM(Small_LLM_Model())
    except Exception as error:
        print(error, "occured while initializing program")
        return False
    try:
        future_json = []
        for prompt in prompts_list:
            act_call = FunctionCall(llm, function_defs, prompt)
            print(f"resolving prompt : {prompt}")
            act_call.find_fn_name()
            act_call.get_param()
            future_json.append(act_call.to_dict())
        print("result :\n" + "\n\n".join("".join('\n'.join(
            [f'[{k} : {v}]' for k, v in dic.items()])) for dic in future_json))
        generate_json_output(future_json, path.output)
    except Exception as error:
        print(error, "occured while running program")
    return bool(path.show_time == "true")


if __name__ == "__main__":
    show_stat = main()
    if show_stat:
        stats = monitor_time.get_all_stats()
        print("\n")
        for k, v in format_dict(stats,
                                *stats,
                                sort_key=lambda stat: stat[1]["total_time"],
                                value_filter=lambda stat: stat["call_count"]):
            print(k, v)
