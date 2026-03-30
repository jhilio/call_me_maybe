from transformers import AutoModelForCausalLM, AutoTokenizer
from sys import stderr

class QwenChatbot:
    def __init__(
            self,
            model_path="/home/aspenle/Documents/projets/python/call_me_maybe/model/qwen"):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True)
        self.history = []

    def generate_response(self, user_input: str) -> str:
        messages = self.history + [{"role": "user", "content": user_input}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt")
        response_ids = self.model.generate(**inputs, max_new_tokens=32768)[0][len(inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=False)
        # Update history
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})

        return response

def process_input(input: str):
    if input.find("file=") != -1:
        a = input.find("file=")
        file_name = input[a+5:].split()[0]
        with open(file_name, "r") as file:
            input = input[0:a] + file.read() + input[a+5+len(file_name):]
        print(input)
    return input


# Example Usage
if __name__ == "__main__":
    chatbot = QwenChatbot()

    # First input (without /think or /no_think tags, thinking mode is enabled by default)
    with open("history_message", "w") as log:
        pass
    for i in range(100):
        try:
            user_input = input("you :")
        except EOFError:
            break
        with open("history_message", "a+") as log:
            print(user_input, file=log)
        user_input = process_input(user_input)
        whole_response = chatbot.generate_response(user_input)
        thinking = whole_response.split("</think>")[0] + "</think>"
        response = whole_response.split("</think>")[1]
        print(f"{thinking}", file=stderr)
        print(f"Bot: {response}")
        print("----------------------")
