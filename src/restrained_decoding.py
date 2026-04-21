from llm_interaction import MyLLM
from math import exp
from utils import monitor_time
import random
import re


def find_closing_quote_in_text(decoded_text: str) -> tuple[bool, int]:
    """
    Returns True if a quote is followed by a space or newline.
    """
    for match in re.finditer(r"[\"']", decoded_text):
        idx = match.end()  # position right after the quote
        if idx < len(decoded_text) and decoded_text[idx] in (" ", "\n"):
            return (True, idx)
    return (False, 0)


@monitor_time
def get_compatible_next_tokens(
        current_output: list[int],
        allowed_token_phrases: list[list[int]]) -> list[int]:
    # returns set of token IDs allowed as the next token
    compatible_next_tokens = set()
    len_output = len(current_output)
    for seq in allowed_token_phrases:
        if len(seq) <= len_output:
            continue
        if seq[0:len_output] == current_output:
            compatible_next_tokens.add(seq[len_output])
    return list(compatible_next_tokens)


@monitor_time
def phrase_only_rd(
        prompt: str,
        allowed: list[str],
        llm: MyLLM,
        temperature: float = 0.7,
        acceptable_margin: float = 0.5,
        max_token: bool = False,
        verbose: bool = False) -> str:
    allowed_token_ids = [llm.encode(s).tolist()[0] for s in allowed]
    input_token = llm.encode(prompt).tolist()[0]
    current_output: list[int] = []
    while (allowed_next := get_compatible_next_tokens(
                current_output, allowed_token_ids)):
        input_ids = input_token + current_output
        logits = llm.llm.get_logits_from_input_ids(input_ids)
        raw_weights = [logits[t] for t in allowed_next]
        # ---- MAX TOKEN PATH (NO PROBS NEEDED) ----
        if max_token:
            next_token = allowed_next[raw_weights.index(max(raw_weights))]
        # ---- SAMPLING PATH (ONLY HERE WE COMPUTE PROBS) ----
        else:
            max_logit = max(raw_weights)
            weights = [exp((w - max_logit) / temperature) for w in raw_weights]
            total = sum(weights)
            probs = [w / total for w in weights]
            sorted_probs = sorted(probs, reverse=True)
            margin = (sorted_probs[0] - sorted_probs[1]
                      if len(sorted_probs) > 1 else 1.0)
            if verbose:
                print(f"{margin=}")
            if margin < acceptable_margin:
                return "None"
            next_token = random.choices(list(allowed_next), weights=weights)[0]
        current_output.append(next_token)
    return llm.decode(current_output)


@monitor_time
def param_fill_rd(
        prompt: str,
        llm: MyLLM,
        max_len: int = 10,
        focus_text: dict[str, float] = {},
        boost_tokens: list[list[int]] | None = None,
        max_token: bool = True,
        verbose: bool = False) -> str:
    """
    Extract text deterministically from LLM logits.

    Args:
        prompt: Full system/user prompt
        llm: Your LLM object
        max_len: Maximum tokens to decode
        focus_text: Optional string to bias token selection toward
    """
    input_ids = llm.encode(prompt).tolist()[0]
    current_output: list[int] = []
    current_text = ""
    focus_ids: dict[int, float] = {}
    # compute bias tokens only from focus_text
    for text, value in focus_text.items():
        for tok in set(llm.encode(text).tolist()[0]):
            focus_ids[tok] = focus_ids.get(tok, 0) + value
    for _ in range(max_len):
        logits = llm.llm.get_logits_from_input_ids(input_ids + current_output)
        # bias logits toward focus_text tokens
        for token_id, boost_value in focus_ids.items():
            logits[token_id] *= boost_value
        if boost_tokens:
            boost_strength = 35
            for phrase in boost_tokens:
                if not current_output:
                    logits[phrase[0]] += boost_strength
                else:
                    last_token = current_output[-1]
                    if last_token in phrase:
                        idx = phrase.index(last_token)
                        if idx < len(phrase) - 1:
                            next_token = phrase[idx + 1]
                            logits[next_token] += boost_strength
        logits[llm.encode("<think>").tolist()[0][0]] = float("-inf")
        logits[llm.encode("</think>").tolist()[0][0]] = float("-inf")
        if not current_output:
            logits[llm.encode("\n").tolist()[0][0]] = float("-inf")
            logits[llm.encode("\n\n").tolist()[0][0]] = float("-inf")
            logits[llm.encode("\n\n\n").tolist()[0][0]] = float("-inf")
            logits[llm.encode("<|endoftext|>").tolist()[0][0]] = float("-inf")
            logits[llm.encode("<|im_end|>").tolist()[0][0]] = float("-inf")
        if max_token:
            next_token = logits.index(max(logits))
        else:
            max_logit = max(logits)
            weights = [exp(w - max_logit) for w in logits]
            next_token = random.choices(
                population=list(range(len(logits))),
                weights=weights)[0]
        current_output.append(next_token)
        current_text = llm.decode(current_output)
        if ((quote := find_closing_quote_in_text(current_text))[0]
                or "\n\n" in current_text
                or "<|im_end|>" in current_text
                or "<|endoftext|>" in current_text):
            current_text = current_text.replace("<|im_start|>", "")
            current_text = current_text.replace("<|im_end|>", "")
            current_text = current_text.replace("<|endoftext|>", "")
            current_text.strip("\n")
            if verbose:
                print(f"break with {current_text}, {quote=}")
            if quote[0]:
                current_text = current_text[0:quote[1]]
            break
    return current_text


@monitor_time
def restrained_decoding_number(
    prompt: str,
    allowed_numbers: list[str],
    llm: MyLLM,
) -> str:

    allowed_token_seqs = [llm.encode(t).tolist()[0] for t in allowed_numbers]
    input_token = llm.encode(prompt).tolist()[0]
    current_output: list[int] = []
    while (allowed_next := get_compatible_next_tokens(
            current_output, allowed_token_seqs)):
        input_ids = input_token + current_output
        logits = llm.llm.get_logits_from_input_ids(input_ids)
        raw_weights = [logits[t] for t in allowed_next]
        # GREEDY: no softmax, no probs
        next_token = allowed_next[raw_weights.index(max(raw_weights))]
        current_output.append(next_token)
        if current_output in allowed_token_seqs:
            break
    return llm.decode(current_output)


@monitor_time
def free_commentary(
        prompt: str,
        llm: MyLLM,
        max_len: int = 10,
        focus_text: dict[str, float] = {},
        max_token: bool = False) -> str:
    """
    Extract text deterministically from LLM logits.

    Args:
        prompt: Full system/user prompt
        llm: Your LLM object
        max_len: Maximum tokens to decode
        focus_text: Optional string to bias token selection toward
    """
    # print(prompt)
    input_ids = llm.encode(prompt).tolist()[0]
    current_output: list[int] = llm.encode("A. the problem").tolist()[0]
    current_text = "A. the problem"
    focus_ids: dict[int, float] = {}
    # compute bias tokens only from focus_text
    for text, value in focus_text.items():
        for tok in set(llm.encode(text).tolist()[0]):
            focus_ids[tok] = focus_ids.get(tok, 0) + value
    for _ in range(max_len):
        logits = llm.llm.get_logits_from_input_ids(input_ids + current_output)
        # bias logits toward focus_text tokens
        for token_id, boost_value in focus_ids.items():
            logits[token_id] *= boost_value
        for i, token in enumerate(current_output):  # dont repeat rule
            logits[token] *= (0.85 * (i / len(current_output)))
        logits[llm.encode("<think>").tolist()[0][0]] = float("-inf")
        logits[llm.encode("</think>").tolist()[0][0]] = float("-inf")
        if not current_output:
            logits[llm.encode("\n").tolist()[0][0]] = float("-inf")
            logits[llm.encode("\n\n").tolist()[0][0]] = float("-inf")
            logits[llm.encode("\n\n\n").tolist()[0][0]] = float("-inf")
            logits[llm.encode("<|endoftext|>").tolist()[0][0]] = float("-inf")
            logits[llm.encode("<|im_end|>").tolist()[0][0]] = float("-inf")
        if max_token:
            next_token = logits.index(max(logits))
        else:
            max_logit = max(logits)
            weights = [exp(w - max_logit) for w in logits]
            next_token = random.choices(
                population=list(range(len(logits))),
                weights=weights)[0]
        current_output.append(next_token)
        current_text = llm.decode(current_output)
        if "<|im_end|>" in current_text or "<|endoftext|>" in current_text:
            current_text.removesuffix("<|im_end|>")
            current_text = current_text.replace("<|endoftext|>", "")
            current_text.strip()
            break
    # print(current_output, current_text)
    return current_text
