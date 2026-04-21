*This project has been created as part of the 42 curriculum by aspenle*

# Call_me_maybe

## Description
This is a small function-calling system using Qwen3 to format demands into function calls.

- **Goal:** transform human written text into strictly formatted function call.
- **Overview:** read files as input to know which functions are available, and a list of prompts to test. With constrained decoding, the program forces Qwen3 to respond only with a valid function name, then, with the help of some prompt and function description, it fills in the arguments.

## Instructions
Use `uv sync` or `make install` to install all dependencies.

Then launch with:
```bash
uv python run src [option]
```

or
```bash
make run
```

Available options:
- `--functions_definition func_def_path` → default: `data/input/functions_definition.json`
- `--input your_prompt_list_file_path` → default: `data/input/function_calling_tests.json`
- `--output output_file_path` → default: `data/output/function_calling_results.json`

Bonus:
- `--show_time` displays execution time metrics at the end of the program

## Resources
All kinds of things on the internet, and a bit of Qwen documentation:
https://qwen.readthedocs.io/en/latest/

AI was used mainly as a search engine; it did not produce any directly copy-pasted code, although it helped a lot with the reconstruction of the encoder.

## Algorithm explanation
1. First, I task the model with selecting the corresponding function.  
   To do that, I list all valid tokens that could start or continue at least one function name.  
   Then I take the most likely candidate and repeat until a full function name is created.

2. Then, with the help of the function description and parameter names/types, I prompt the model with heavy tweaking of logits values (boost tokens present in the prompt, nerf function names and parameter names, etc.).

3. Another system I made is that after discovering a candidate for a parameter, I test it with some hardcoded tests + the model judges its own creation. If any test fails, it comments on the failure and retries generating arguments using its own comment as context.

## Design decisions
I decided to keep everything simple. One big class `FunctionCall` handles generating its arguments with the help of functions from `restrained_decoding.py`.

The main script simply reads arguments, then reads the input file to set up the main loop iterating over all test prompts. The program finishes by creating the output file with all gathered data.

## Performance analysis
After a lot of tweaking in prompts and logits bias, I think I achieved relatively good accuracy. The only speed problem can occur if many parameter candidates are judged invalid multiple times in a row, which is rare, so performance should not be a problem.

## Challenges faced
A lot of them, but the biggest one was knowing when to stop generating text when a string parameter forces the model to generate freely. To solve this problem, I found multiple solutions that work well together.

### Solutions
1. In the prompt, I tell the model to finish with empty lines; if detected, I stop generation.
2. I start the prompt with a quote if the model terminates the quote, I stop generation.
3. If the LLM generates a `<|im_end|>` or `<|end_of_text|>` token, I stop generation.

With these three end conditions, the LLM is quite good at knowing when its work is done.

## Testing strategy
I continuously run the program with given test exercises and prompts of my own, and sometimes with functions I created myself.

## Example usage
```bash
uv sync
uv run python src [flag for file emplacement if necessary]
```