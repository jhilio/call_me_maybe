from utils import monitor_time
import json
import torch
from torch import Tensor
import regex as re
import unicodedata
class MyLLM:
    def __init__(self, llm):
        self.llm = llm

        self.byte_encoder = self.bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        # -----------------------------
        # Regex from tokenizer.json
        # -----------------------------
        self.ansi_regex = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        self.pattern = re.compile(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
        )
        # -----------------------------
        # Load tokenizer JSON
        # -----------------------------
        with open(llm.get_path_to_tokenizer_file()) as f:
            tok = json.load(f)

        self.vocab = tok["model"]["vocab"]
        self.id_to_token = {v: k for k, v in self.vocab.items()}

        # -----------------------------
        # Added tokens (HF priority)
        # -----------------------------
        self.added_tokens = sorted(
            [(t["content"], t["id"]) for t in tok["added_tokens"]],
            key=lambda x: len(x[0]),
            reverse=True
        )
        self.id_to_token.update({v: k for k, v in self.added_tokens})

        # -----------------------------
        # Merges (BPE)
        # -----------------------------
        with open(llm.get_path_to_merges_file()) as f:
            merges = f.read().splitlines()

        self.merges = {tuple(m.split()): i for i, m in enumerate(merges)}


    # =========================================================
    # BYTE LEVEL (HF equivalent)
    def bytes_to_unicode(self):
        bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        cs = bs[:]

        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1

        cs = [chr(c) for c in cs]
        print(bs, "\n", cs)
        return dict(zip(bs, cs))

    def byte_level_encode(self, text: str) -> str:
        return "".join(self.byte_encoder[b] for b in text.encode("utf-8"))

    def byte_level_decode(self, text: str) -> str:
        byte_array = bytearray([self.byte_decoder[c] for c in text])
        return byte_array.decode("utf-8", errors="replace")

    # =========================================================
    # SPECIAL TOKEN MATCHING (HIGHEST PRIORITY)
    # =========================================================
    def split_special_tokens(self, text: str):
        """
        Returns list of:
        (is_special: bool, value: str | int)
        """
        i = 0
        n = len(text)
        out = []

        while i < n:
            matched = False

            for tok, tid in self.added_tokens:
                if text.startswith(tok, i):
                    out.append((True, tid))
                    i += len(tok)
                    matched = True
                    break

            if matched:
                continue

            out.append((False, text[i]))
            i += 1
        return out

    # =========================================================
    # GROUP NORMAL TEXT INTO STRINGS
    # =========================================================
    def group_text(self, spans):
        """
        merges consecutive normal chars into strings
        keeps special tokens separate
        """
        out = []
        buffer = ""
        for is_special, val in spans:
            if not is_special:
                buffer += val
            else:
                if buffer:
                    out.append((False, buffer))
                    buffer = ""
                out.append((True, val))
        if buffer:
            out.append((False, buffer))

        return out

    # =========================================================
    # NORMAL TOKENIZATION PIPELINE (regex + bytelevel)
    # =========================================================
    def normal_tokenize(self, text: str):
        text = text.replace("\u00a0", " ")
        return [self.byte_level_encode(p) for p in self.pattern.findall(text)]
    # =========================================================
    # BPE MERGE
    # =========================================================
    def bpe(self, tokens: list[str]) -> list[str]:
        while True:
            best_pair = None
            best_rank = float("inf")
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.merges.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break
            i = 0
            while i < len(tokens) - 1:
                if (tokens[i], tokens[i + 1]) == best_pair:
                    tokens[i:i+2] = [tokens[i] + tokens[i + 1]]
                else:
                    i += 1

        return tokens

    # =========================================================
    # FULL TOKENIZATION
    # =========================================================
    def tokenize(self, text: str):
        spans = self.split_special_tokens(text)
        grouped = self.group_text(spans)
        output = []
        for is_special, val in grouped:
            if is_special:
                output.append(val)  # already token id
                continue

            for part in self.normal_tokenize(val):
                tokens = list(part)
                merged = self.bpe(tokens)
                output.extend(self.vocab[t] for t in merged)
        return output

    # =========================================================
    # ENCODE
    # =========================================================
    def clean_ansi(self, text: str):
        return self.ansi_regex.sub('', text)

    @monitor_time
    def encode(self, text: str) -> Tensor:
        text = unicodedata.normalize("NFC", text)
        text = self.clean_ansi(text)
        text1 = torch.tensor([self.tokenize(text)], dtype=torch.long)
        return text1

    # =========================================================
    # DECODE
    # =========================================================
    def decode(self, ids: list[int] | Tensor) -> str:
        if isinstance(ids, Tensor):
            ids = ids.tolist()[0]
        return self.byte_level_decode(
            "".join(self.id_to_token[i] for i in ids))

#         text2 = self.llm.encode(text)
#         if str(text1.tolist()) != str(text2.tolist()):
#             print(f"""error 1:
# \033[30m{text}\033[0m
# \033[31m{text1}\033[0m
# \033[32m{text2}\033[0m""")
#             print(f"""
# O \033[31m|{"|".join(self.decode([a]) for a in text1.tolist()[0])}\033[0m
# # \033[32m|{"|".join(self.decode([a]) for a in text2.tolist()[0])}\033[0m
# """)