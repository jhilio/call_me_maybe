from hmac import new

from llm_sdk import Small_LLM_Model
from math import exp
import random

def get_compatible_next_tokens(current_output, allowed_token_phrases):
    # returns set of token IDs allowed as the next token
    compatible_next_tokens = set()
    len_output = len(current_output)
    for seq in allowed_token_phrases:
        if len(seq) <= len_output:
            continue
        if seq[0:len_output] == current_output:
            compatible_next_tokens.add(seq[len_output])
    return list(compatible_next_tokens)


def phrase_only_rd(
        prompt: str,
        allowed: list[str],
        llm: Small_LLM_Model,
        temperature: int=0.7,
        acceptable_margin=0.5,
        max_token=False,
        verbose=False) -> str:
    allowed_token_ids = [llm.encode(string).tolist()[0] for string in allowed]
    input_token = llm.encode(prompt).tolist()[0]
    current_output = []
    if verbose:
        print(prompt)
    while (allowed_next := get_compatible_next_tokens(
        current_output, allowed_token_ids)):
        if not allowed_next:
            break
        input_ids = input_token + current_output
        logits = llm.get_logits_from_input_ids(input_ids)
        raw_weights = [logits[token_id] for token_id in allowed_next]
        # numerically stable softmax
        max_logit = max(raw_weights)
        weights = [exp((w - max_logit) / temperature) for w in raw_weights]
        # normalize
        total = sum(weights)
        probs = [w / total for w in weights]

        sorted_probs = sorted(probs, reverse=True)

        if len(sorted_probs) > 1:
            margin = sorted_probs[0] - sorted_probs[1]
        else:
            margin = 1.0  # only one option
        if margin < acceptable_margin:   # tune this (0.1–0.3 usually good)
            return "None"
        if max_token:
            next_token = allowed_next[weights.index(max(weights))]
        else:
            next_token = random.choices(population=list(allowed_next), weights=weights)[0]
        current_output.append(next_token)
    return llm.decode(current_output)


def free_text_rd(
        prompt: str, llm, max_len: int = 10,
        focus_text: str | None = None,
        boost_tokens: list[int] | None = None,
        acceptable_margin=0.2):
    """
    Extract text deterministically from LLM logits.
    
    Args:
        prompt: Full system/user prompt
        llm: Your LLM object
        max_len: Maximum tokens to decode
        focus_text: Optional string to bias token selection toward
    """
    input_ids = llm.encode(prompt).tolist()[0]
    current_output = []

    # compute bias tokens only from focus_text
    if focus_text:
        focus_ids = set(llm.encode(focus_text).tolist()[0])
    else:
        focus_ids = set(input_ids)
    # optional: boost special constants
    # print(prompt)
    for _ in range(max_len):
        logits = llm.get_logits_from_input_ids(input_ids + current_output)
        # bias logits toward focus_text tokens
        for token_id in focus_ids:
            logits[token_id] += 2.5
        if boost_tokens:
            for token_id in boost_tokens:
                logits[token_id] += 10  # stronger boost
        # deterministic argmax
        next_token = max(range(len(logits)), key=lambda i: logits[i])
        current_output.append(next_token)
        current_text = llm.decode(current_output)
        if "\n" in current_text or current_text.count("'") > 1 or  current_text.count('"') > 1:
            break
    return current_text

def restrained_decoding_number(
    prompt: str,
    allowed_numbers: list[str],
    llm: Small_LLM_Model,
    temperature: float = 0.01,
) -> str:
    """
    Generate multiple tokens from the LLM using a restricted set of allowed tokens.
    Stops only when no compatible next token is available or max_len is reached.
    """
    # encode the allowed tokens
    allowed_token_seqs = [llm.encode(t).tolist()[0] for t in allowed_numbers]
    input_token = llm.encode(prompt).tolist()[0]
    current_output = []

    while (allowed_next := get_compatible_next_tokens(current_output, allowed_token_seqs)):
        input_ids = input_token + current_output
        logits = llm.get_logits_from_input_ids(input_ids)
        raw_weights = [logits[token_id] for token_id in allowed_next]

        # softmax
        max_logit = max(raw_weights)
        weights = [exp((w - max_logit)/temperature) for w in raw_weights]
        total = sum(weights)
        probs = [w/total for w in weights]

        # pick the most likely token
        next_token = allowed_next[probs.index(max(probs))]
        current_output.append(next_token)

        # stop if we generated a complete number
        if any(current_output == seq for seq in allowed_token_seqs):
            break
    return llm.decode(current_output)





# def restrained_decoding_string(prompt: str, llm, max_len: int = 50, temperature: float = 0.5, previous_strings=None) -> str:
#     """
#     Token-by-token string decoding:
#     - Constrains to prompt substrings when possible
#     - Allows free generation otherwise
#     """
#     previous_strings = previous_strings or []
#     input_ids = llm.encode(prompt).tolist()[0]
#     current_output = []

#     prompt_tokens = llm.encode(prompt).tolist()[0]

#     for _ in range(max_len):
#         logits = llm.get_logits_from_input_ids(input_ids + current_output)

#         # Determine compatible next tokens
#         # 1. Tokens that continue a substring of the prompt
#         compatible_next = []
#         prefix_len = len(current_output)
#         for i in range(len(prompt_tokens)):
#             if prompt_tokens[i:i+prefix_len] == current_output[-prefix_len:]:
#                 if i + prefix_len < len(prompt_tokens):
#                     compatible_next.append(prompt_tokens[i+prefix_len])

#         # 2. If no compatible_next, fallback to all vocab
#         allowed_next = compatible_next if compatible_next else list(range(len(logits)))

#         # Softmax sampling
#         max_logit = max([logits[t] for t in allowed_next])
#         weights = [exp((logits[t]-max_logit)/temperature) for t in allowed_next]
#         total = sum(weights)
#         probs = [w/total for w in weights]

#         next_token = random.choices(allowed_next, weights=probs)[0]
#         decoded_next = llm.decode([next_token])

#         # Stop if sentence-ending token
#         if any(c in ('.', '?', '!', '\n') for c in decoded_next):
#             break

#         current_output.append(next_token)

#         # Avoid duplicating previously extracted strings
#         current_str = llm.decode(current_output)
#         if current_str in previous_strings:
#             break

#     return llm.decode(current_output)