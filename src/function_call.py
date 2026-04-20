

from typing import Any
from llm_interaction import MyLLM
from restrained_decoding import free_commentary, phrase_only_rd, param_fill_rd, restrained_decoding_number
from utils import monitor_time
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
    "number": ("""
<|im_start|>system
Task: fill the parameter with the correct numeric value from the user request

<tools>
{func}
</tools>
Current parameter state: {param_state}
Parameter to fill: {param_name}
Expected type: number

<|im_start|>user
{prompt}<|im_end|>
"""
    ),
    "integer":("""
<|im_start|>system
Task: Extract an integer value from the user request.

<tools>
{func}
</tools>
Current parameter state: {param_state}
Parameter to fill: {param_name}
Expected type: integer

<|im_start|>user
{prompt}<|im_end|>
"""),
    "boolean": ("""
<|im_start|>system
You are extracting a boolean parameter for a function.

<tools>
{func}
</tools>
Actual parameter_state: {param_state}
Parameter name you're filling: {param_name}
Expected type: boolean

Fill the parameter according to function definition and the user request.<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
"""
    ),
    "string": ("""
<|im_start|>system
You are a function calling LLM and your task is to write the parameter value of a function depending on user input

function definition : {func[description]}
Param_state: {param_state}
Parameter you are filling: {param_name}

Rules:
- DO NOT execute the function.
- Do NOT repeat previous parameter values.<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
"""
    ),
}
    COMMENTARY_PROMPT = ("""
<|im_start|>system
Context:
'{param_try}' is invalid for '{param_name}' in this function :
<tools>
{func}
</tools>
from prompt '{prompt}'.
here is the curent parameter state;
<tools>
{param_state}
</tools>
Do not change prompt casing.
Param value should never be repeated
Explain consisely (30 to 50 character):
A. Why it's invalid
B. How to fix it<|im_end|>
<|im_start|>assistant
"""
                    )
    def __init__(self, llm: MyLLM, func_data: list[dict], prompt: str):
        self.function_name = "None" 
        self.parameter: dict[str, Any] = {}
        self.llm = llm
        self.func_data = func_data
        self.prompt = prompt
        REGEX = {
            "vowels": ["[aeiou]\n\n"],
            "numbers": ["-?\\d+\n\n"]
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
        self.all_bias = BIASED
    
    @monitor_time
    def find_fn_name(
            self,
            temperature: float=0.7) -> None:
        allowed: list[str] = [str(item.get("name")) for item in self.func_data if "name" in item]
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
                base_prompt + self.prompt + post_prompt,
                allowed,
                self.llm,
                temperature,
                acceptable_margin=0.2, max_token=True)
        else:
            self.function_name = "None"

    @monitor_time
    def get_param(self) -> None:
        MAX_TRY = 3
        if self.function_name == "None":
            return
        try:
            func = [func for func in self.func_data if func["name"] == self.function_name][0]
        except IndexError:
            self.parameter = {}
            return
        param_schema = func.get("parameters", {})
        self.parameter = {a: "" for a in param_schema}
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
                full_prompt = base_prompt + retry_context + "<|im_start|>assistant\n"

                param_try = self.set_param(param_type, param_name, full_prompt, random=bool(retry_context))
                if self.judge_param(param_try, param_name, func):
                    break
                if i < MAX_TRY - 1:
                    print(f"temporary result : {param_try}")
                    act_rompt = self.COMMENTARY_PROMPT.format(
                            param_try=param_try, param_name=param_name, func=func,prompt=self.prompt, param_state=self.parameter)
                    retry_context = "advice from previous attempt : \n" + free_commentary(
                        act_rompt,
                        self.llm,
                        focus_text={
                            act_rompt: 0.2
                        },
                        max_token=False,
                        max_len=40
                    ).strip("\n") + "\n\n"
                    print(f"\n\n{retry_context=}")
            self.parameter[param_name] = param_try
            print(f"validated parameter |{param_name}| to be |{param_try}|")
            
    def judge_param(self, param_value: Any, param_name: str, func) -> bool:
        if param_value is None:
            return False
        judge_prompt = f"""
<|im_start|>system
Task: judge if a candidate value fit a given parameter to a function call
<tools>
{func}
</tools>
here is the curent parameter state;
<tools>
{self.parameter}
</tools>
and the current candidate for {param_name} : 
{param_value}.
respond with :
"A" if the candidate EXACTLY fit the user demande
"B" if this candidate is a repeat of previous parameter or doenst fit the user request<|im_end|>
<|im_start|>user
{self.prompt}<|im_end|>
<|im_start|>assistant
"""

        if phrase_only_rd(judge_prompt
            , ["A", "B"], self.llm, max_token=True) == "B":
            print(f"{judge_prompt}\n\n\n\n\nrejected by ai : {param_value}")
            return False
        if param_value in self.parameter.values():
            return False
        if param_value in self.parameter.keys():
            return False
        return True

    def set_param(self,
                  param_type: str,
                  param_name: str,
                  full_prompt: str,
                  random: bool=False) -> str | bool | int | float | None:
        result: str | None | float | int | bool = None
        if param_type == "number":
            numbers_in_prompt = [x for x in extract_numbers(self.prompt, allow_float=True) if float(x) not in self.parameter.values()]
            value_output = restrained_decoding_number(full_prompt, numbers_in_prompt, self.llm)
            try:
                result = float(value_output)
            except Exception:
                result = None
        elif param_type == "integer":
            numbers_in_prompt = extract_numbers(self.prompt, allow_float=True)
            typed_number = [float(x) if '.' in x else int(x) for x in numbers_in_prompt]
            left_in_prompt = [str(x) for x in typed_number if x not in self.parameter.values() and isinstance(x, int)]
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
                    "  ".join(self.function_name.split("_-")): 0.4,
                    self.function_name: 0.4,
                    self.function_name.lower(): 0.4,
                    param_name: 0.1,
                    # "  ".join([v for k, v in self.parameter.items()]): 0.01, 
                    " Okay": 0.5,
                    " okay": 0.5,
                    "Okay": 0.5,
                    "okay": 0.5,
                    param_name.lower(): 0.1
                }
            # print (focus)
            for name, bias in self.all_bias.items():
                if param_name == name:
                    for k, v in bias.items():
                        if k in self.prompt:
                            for phrases in v:
                                boosted.append(self.llm.encode(phrases).tolist()[0])
            final_prompt = full_prompt + ('"' if "'" in self.prompt else "'")
            value_output = param_fill_rd(
                final_prompt,
                self.llm,
                max_len=20,
                focus_text=focus,
                boost_tokens=boosted,
                max_token=not random
                )
            result = value_output.strip(" \n")
            while result:
                if (result[0] == '"' or final_prompt[-1] == '"') and result[-1] == '"':
                    result = result.strip('"')
                elif (result[0] == "'" or final_prompt[-1] == "'") and result[-1] == "'":
                    result = result.strip("'")
                else:
                    break

        return result

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "func_name": self.function_name,
            "parameters": self.parameter
        }
