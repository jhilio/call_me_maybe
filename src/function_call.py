

from typing import Any
from llm_sdk import Small_LLM_Model
from restrained_decoding import free_commentary, phrase_only_rd, param_fill_rd, restrained_decoding_number
import re

def extract_numbers(prompt: str, allow_float: bool=False) -> list[str]:
    if allow_float:
        return re.findall(r"-?\d+\.?\d*", prompt)
    else:
        return re.findall(r"(?<![\d.])-?\d+(?!\.\d)", prompt)

def extract_words(text: str) -> list[str]:
    # Match sequences of letters, digits, or underscores
    return re.findall(r"\b\w+\b", text)

class FunctionCall:

    param_type_prompts = {
    "number": (
    "Task: Extract a numeric value from the user request.\n\n"

    "Function: {func}\n"
    "User request: {prompt}\n"
    "Current parameter state: {param_state}\n"
    "Parameter to fill: {param_name}\n"
    "Expected type: number\n\n"

    "Rules:\n"
    "- Extract exactly ONE number from the user request.\n"
    "- If multiple numbers are present, choose the FIRST number from left to right.\n"
    "- If this number is already used in parameter_state, choose the next unused number.\n"
    "- NEVER evaluate a result, simply extract values already in user request\n"
    ),
    "boolean": (
        "You are extracting a boolean parameter for this function: {func}\n"
        "User request: {prompt}\n"
        "Actual parameter_state: {param_state}\n"
        "Parameter name you're filling: {param_name}\n"
        "Expected type: boolean\n"
        "fill the parameter according to function definition and the user request\n\n"
    ),
    "string": (
        "Function: {func[name]}\n"
        "Description: {func[description]}\n"
        "Param_state: {param_state}\n"
        "Parameter: {param_name}\n"
        "Type: string\n"
        "Request: |\33[31m{prompt}\33[0m|\n\n"
        "Rules :\n"
        "   Do NOT apply the function.\n"
        "   Do NOT repeat previous parameter values.\n"
        "   Finish your response by a new line.\n\n"
    ),
}
    def __init__(self, llm: Small_LLM_Model, func_data: list[dict], prompt: str):
        self.function_name = "None" 
        self.parameter = {}
        self.llm = llm
        self.func_data = func_data
        self.prompt = prompt
    
    def find_fn_name(
            self,
            temperature: int=0.7):
        allowed = [item.get("name") for item in self.func_data if "name" in item]
        base_prompt = (
            "You are a strict function router.\n"
        "Your job is to select the correct function from the list below.\n\n"
        f"{str(self.func_data)}\n\n"
        "Rules:\n"
        "- Regex can do a lot of thing dont hesitate to chose it"
        "- Return the function name ONLY if it clearly and exactly matches the user request.\n"
        "- If there is any ambiguity, mismatch, or missing function, return 'None'.\n"
        "- Most requests do NOT match any function.\n"
            )
        post_prompt = "Function router:"
        if self.prompt.strip():
            self.function_name = phrase_only_rd(
                base_prompt + self.prompt + post_prompt, allowed, self.llm, temperature, acceptable_margin=0.1, max_token=True)
        else:
            self.function_name = "None"

    def get_param(self):
        MAX_TRY = 5
        try:
            func = [func for func in self.func_data if func["name"] == self.function_name][0]
        except IndexError:
            self.parameter = {}
            return
        param_schema = func.get("parameters", {})
        self.parameter = {}

        for param_name, param_info in param_schema.items():
            param_type = param_info.get("type", "string")

            prompt_template = self.param_type_prompts.get(param_type, self.param_type_prompts["string"])
            base_prompt = prompt_template.format(
                func=func,
                param_state=self.parameter,
                prompt=self.prompt,
                param_name=param_name
            )
            retry_context = ""
            i = 0
            for i in range(MAX_TRY):
                full_prompt = base_prompt + retry_context + "parameter value:"
                # print(full_prompt)
                param_try = self.set_param(param_type, param_name, full_prompt, random=bool(retry_context))
                if self.judge_param(param_try):
                    break
                if i < MAX_TRY - 1:
                    commentary_prompt = (
                        "Context : \n"
                        f"the value '{param_try}' is not a valid output for\n"
                        f"parameter '{param_name}' in the function {func} for\n"
                        f"the user prompt '{self.prompt}'\n\n"
                        
                        "your task is to descibe :\n"
                        "A, why the value dont fit the argument needed for the user request \n"
                        "B, quick advice on how to fix it\n"
                        "Commentary: "
                    )
                    retry_context = "advice from previous attempt : \n" + free_commentary(
                        commentary_prompt,
                        self.llm,
                        max_len=80
                    ).strip("\n") + "\n\n"
                    print(f"\n\n\n\n\n{retry_context=}")
            self.parameter[param_name] = param_try
            print(f"validated parameter {param_name} to be {param_try}")
            
    def judge_param(self, param_value: Any):
        if param_value is None:
            return False
        if isinstance(param_value, str) and param_value.strip().lower() == self.prompt.strip().lower():
            return False
        if param_value in self.parameter.values():
            return False
        return True

    def set_param(self, param_type, param_name, full_prompt, random: bool=False):
        REGEX = {
            "vowels": ["[aeiou]\n"],
            "numbers": ["-?\\d+"]
        }
        REPLACEMENT = {
            "asterisk": ["*\n"],
            "numbers": ["NUMBERS"]
        }
        QUERY = {
            "query": ["SELECT",
                  "*",
                  "FROM"]
        }
        BIASED = {"regex": REGEX, "replacement": REPLACEMENT, "SQL": QUERY}

        result = None
        if param_type == "number":
            numbers_in_prompt = [x for x in extract_numbers(self.prompt, allow_float=True) if float(x) not in self.parameter.values()]
            value_output = restrained_decoding_number(full_prompt, numbers_in_prompt, self.llm)
            try:
                result = float(value_output)
            except Exception:
                result = None
        elif param_type == "integer":
            numbers_in_prompt = extract_numbers(self.prompt, allow_float=True)
            numbers_in_prompt = [float(x) if '.' in x else int(x) for x in numbers_in_prompt]
            left_in_prompt = [str(x) for x in numbers_in_prompt if x not in self.parameter.values()]
            value_output = restrained_decoding_number(full_prompt, left_in_prompt, self.llm)
            try:
                result = int(value_output)
            except Exception:
                result = None
        elif param_type == "boolean":
            allowed_string = ["true", "false"]
            value_output = phrase_only_rd(full_prompt, allowed_string, self.llm, max_token=True)
            val = value_output.strip().lower()
            if val == "true":
                result = True
            else:
                result = False
        else:  # string or fallback
            boosted = []
            focus = {
                    self.prompt: 1.2,
                    self.function_name: 0.4,
                    self.function_name.lower(): 0.4,
                    param_name: 0.1,
                    param_name.lower(): 0.1
                }
            for name, bias in BIASED.items():
                if param_name == name:
                    for k, v in bias.items():
                        if k in self.prompt:
                            boosted.extend(self.llm.encode(v).tolist()[0])
            value_output = param_fill_rd(
                full_prompt + ('"' if "'" in self.prompt else "'"),
                self.llm,
                max_len=20,
                focus_text=focus,
                boost_tokens=boosted,
                max_token=not random
                )
            result = value_output.strip(" \n'\"")
        return result

    def to_dict(self):
        return {
            "prompt": self.prompt,
            "func_name": self.function_name,
            "parameters": self.parameter
        }
    



#             checks = {
#         "user_request": f"""
# Candidate value '{param_value}' for parameter '{param_name}'.
# User request: "{self.prompt}"

# Choose the correct option:

# A) The candidate correctly satisfies the user request
# B) The candidate does not satisfy the user request

# Answer ONLY "A" or "B".
# Response : 
# """
#         }
#         base_prompt = f"Function :\n{func_def}.\n"
#         results = {}
#         for check_name, prompt in checks.items():
#             output = phrase_only_rd(base_prompt + prompt, allowed_string, self.llm,
#                                     acceptable_margin=0, max_token=True)
#             if output == "A":
#                 output = "pass"
#             if output == "B":
#                 output = "blocked"
#             results[check_name] = output
#             # if output == "blocked" :
#             print(f"{check_name} check for {param_name} [{param_value}]: {output}")

#         # Final decision: must pass all checks
#         if all(v == "pass" for v in results.values()):
#             return True