

from queue import Empty

from llm_sdk import Small_LLM_Model
from restrained_decoding import phrase_only_rd, free_text_rd, restrained_decoding_number
import re

def extract_numbers(prompt: str) -> list[str]:
    return re.findall(r"-?\d+\.?\d*", prompt)

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
    "- Do NOT invent or compute any number.\n"
    "- Only use numbers explicitly written in the request.\n"
    "- If this number is already used in parameter_state, choose the next unused number.\n"
    "- Output only the number, no text.\n"
    "- NEVER evaluate a result, simply extract values already in user request\n"
    ),
    "boolean": (
        "You are extracting a boolean parameter for this function: {func}\n"
        "User request: {prompt}\n"
        "Actual parameter_state: {param_state}\n"
        "Parameter name you're filling: {param_name}\n"
        "Expected type: boolean\n"
        "Only return 'true' or 'false'. Do not add explanations.\n"
    ),
    "string": (
        "Function: {func[name]}\n"
        "Description: {func[description]}\n"
        "Param_state: {param_state}\n"
        "Parameter: {param_name}\n"
        "Type: string\n"
        "Request: {prompt}\n\n"
        "Extract ONLY the original value for this parameter from the request.\n"
        "Do NOT apply the function."
        "Finish with a new line at the end"
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
                base_prompt + self.prompt + post_prompt, allowed, self.llm, temperature, 0.4, True)
        else:
            self.function_name = "None"

    def get_param(self):
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
            result = None
            while not (result := self.judge_param(func, param_name)) and i < 5:
                full_prompt = base_prompt + retry_context + "parameter value:"
                self.set_param(param_type, param_name, full_prompt)
                print(f"try {i}\n", "\n".join(f"{k}, {v}" for k,v in {
                    "result": f"=={self.parameter[param_name]}==",
                    "judgement": result,
                    "context": retry_context
                }.items()), "\n\n")
                if i < 4:
                    commentary_prompt = (
                        f"Provide a short commentary on why the parameter '{param_name}' with "
                        f"value '{self.parameter[param_name]}' might not be a valid output for the user prompt:\n"
                        f"'{self.prompt}' and provide detailed advice on how to fix it\n"
                        "Commentary: "
                    )
                    retry_context = free_text_rd(
                        commentary_prompt,
                        self.llm,
                        max_len=50
                    )
                i = i+1
            
    def judge_param(self, func_def: dict, param_name: str):
        allowed_string = ["yes", "no"]
        param_value = self.parameter.get(param_name, None)
        if param_value is None:
            return False
        value_output = phrase_only_rd(f"does '{param_value}' fit the description of {func_def} parameter {param_name}", allowed_string, self.llm, acceptable_margin=0, max_token=True)
        if value_output == "yes":
            return True
        return False

    def set_param(self, param_type, param_name, full_prompt):
        CONSTANTS = {
            "vowels": "aeiou",
            "asterisk": "*",
            "numbers": "-?\\d+\\.?\\d*"
        }
        if param_type == "number":
            numbers_in_prompt = extract_numbers(self.prompt)
            value_output = restrained_decoding_number(full_prompt, numbers_in_prompt, self.llm)
            try:
                self.parameter[param_name] = float(value_output)
                numbers_in_prompt.remove(value_output)
            except Exception:
                self.parameter[param_name] = None
        elif param_type == "boolean":
            allowed_string = ["true", "false"]
            value_output = phrase_only_rd(full_prompt, allowed_string, self.llm)
            val = value_output.strip().lower()
            if val == "true":
                self.parameter[param_name] = True
            else:
                self.parameter[param_name] = False
        else:  # string or fallback
            boosted = []
            for k, v in CONSTANTS.items():
                if k in self.prompt:
                    boosted.extend(self.llm.encode(v).tolist()[0])
            value_output = free_text_rd(full_prompt, self.llm, max_len=20, focus_text=self.prompt, boost_tokens=boosted)
            self.parameter[param_name] = value_output.strip(" \n'\"")

    def to_dict(self):
        return {
            "prompt": self.prompt,
            "func_name": self.function_name,
            "parameters": self.parameter
        }