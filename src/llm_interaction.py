import json
from torch import Tensor
import torch
from utils import monitor_time
from llm_sdk import Small_LLM_Model
import re

def debug(*stuff, **morestuff):
    for a in stuff:
        print("".join(f"{b}={c} \n" for b, c in a.items()))
    for a in morestuff:
        print(f"{a}")

class MyLLM:

    @monitor_time
    def __init__(self, llm: Small_LLM_Model):
        self.llm: Small_LLM_Model = llm
        self.ansi_regex = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

        with open(self.llm.get_path_to_tokenizer_file(), "r") as tokeniser_file:
            tokeniser = json.load(tokeniser_file)
            added_tokens = tokeniser["added_tokens"]
            self.special_tokens = {
                t["content"]: t["id"]
                for t in added_tokens
            }
            self.sorted_special_tokens = sorted(
                self.special_tokens.items(),
                key=lambda x: len(x[0]),
                reverse=True
            )
            self.tokenizer = {k: v for k, v in tokeniser["model"]["vocab"].items()}
            self.tokenizer.update({k: v for k, v in self.special_tokens.items()})
            self.vocab = {v: k for k, v in tokeniser["model"]["vocab"].items()}
            self.vocab.update({v: k for k, v in self.special_tokens.items()})
        with open(self.llm.get_path_to_merges_file(), "r") as merges_file:
                self.merges_pair = {
                    tuple(line.split()): i
                    for i, line in enumerate(merges_file.read().splitlines())
                }
        translate = {" ": "\u0120",
                    "\t": "\u0109",
                    "\n": "\u010a",
                    "\r": "\u010d",
                    }
        self.translate_table = str.maketrans(translate)
        self.untranslate_table = str.maketrans({v: k for k, v in translate.items()})

    def replace_special_token(self, text: str) -> list[tuple[str, bool]]:
        current = [(text.translate(self.translate_table), False)]
        for token, value in self.sorted_special_tokens:
            next = []
            for parts, spec in current:
                splited = parts.split(token)
                if parts.startswith(token):
                    next.append((str(value), True))
                for i in range(len(splited)-1):
                    next.append((splited[i], spec))
                    next.append((str(value), True))
                next.append((splited[-1], spec))
            current = next
        next = []    
        for text, spec in current:
            if spec:
                next.append((text, True))
            else:
                next.extend([(char, False) for char in text])
        current = next
        return current


    @monitor_time
    def tokenize(self, text: str) -> list[int]:
        symbols = self.replace_special_token(text)
        while True:
            pairs = []
            if len(symbols) < 2:
                break
            for (a, a_spec), (b, b_spec) in zip(symbols, symbols[1:]):
                if a_spec or b_spec:
                    continue
                pairs.append((a, b))
            # find best merge (lowest index in merges file = highest priority)
            best_pair = None
            best_rank = float("inf")
            for p in pairs:
                if p in self.merges_pair and self.merges_pair[p] < best_rank:
                    best_pair = p
                    best_rank = self.merges_pair[p]
            if best_pair is None:
                break
            new_symbols = []
            skip = False
            for (a, a_spec), (b, b_spec) in zip(symbols, symbols[1:]):
                if skip:
                    skip = False
                elif (a, b) == best_pair:
                    new_symbols.append((a + b, False))
                    skip = True
                else:
                    new_symbols.append((a, a_spec))
            if (a, b) != best_pair:
                new_symbols.append((b, b_spec))
            symbols = new_symbols
        final = [self.tokenizer[symbol] if not spec else int(symbol) for symbol, spec in symbols]
        return final 
    
    def clean_ansi(self, text: str):
        return self.ansi_regex.sub('', text)

    # def encode(self, text: str) -> Tensor:



    @monitor_time
    def encode(self, text: str) -> Tensor:
        text = self.clean_ansi(text)
        text1 = torch.tensor([self.tokenize(text)], dtype=torch.long)
        text2 = self.llm.encode(text)
        # if str(text1.tolist()) != str(text2.tolist()):
        #     print(f"error 1: \n\033[33m{text}\033[0m\n\033[32m{text1}\033[0m, \033[034{text2}\033[0m")
        return text2

    def decode(self, ids: Tensor | list[int]) -> str:
        if isinstance(ids, Tensor):
            ids = Tensor.tolist()[0]
        
        text1 = "".join([self.vocab[token] for token in ids]).translate(self.untranslate_table)
        text2 = self.llm.decode(ids)
        # if text1 != text2:
        #     print(f"error 2: \n{ids}\n{text1}, {text2}")
        return text2