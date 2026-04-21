*This project has been created as part of the 42 curriculum by aspenle*


# Call_me_maybe

## Description
    this is a small function calling system using qwen3 to format demand into function call

    - **Goal:** transform human writen text into strictly formated function code.
    - **Overview:** read files as input to know which function is at disposition and a list of promt to test, with constrained decoding the program forces qwen3 to respond only with valid function name, then with help of the prompt and function description it fills in the argument.

## Instructions
    use uv sync or make install to install all dependencies.

    Then launch with:
    uv python run src [option]
    or make run

    available option are : 
    --functions_definition func_def_path to data/input/functions_definition.json
    --input your_prompt_list_file_path : default to data/input/function_calling_tests.json
    --output output_file_path : default to data/output/function_calling_results.json

bonus:
--show_time display metrics about execution time at end of program

## Resources
    all type of thing on the internet, and a bit of qwen documentation:
    https://qwen.readthedocs.io/en/latest/
    AI was used mainly as search engine, it did not produce any untouched copy pasted code althought it helped a lot for the recreation of the encoder



##  Algorithm explanation
    1.
    first i task the Model to the corresponding function, 
    to do that i list all valid token that would start or continue at least 1 function name
    then take the most likely candidate and redo it again until a full function name is created
    2.
    i then with the help of function description and parameter name/type, prompt the model with a lot of tweaking of logits value (boost token present in prompt, nerf the function name and parameter name ect)
    3. another system that i made is after discovering a candidate to a parameter, i test it with some harcoded test + the model judge its own creation, if any test fail it will comment on the failure and retry to generate argument with as context its own comment

## Design decisions
    i decided to keep everything simple, One big class FunctionCall will handle generating its argument with the help of function from 'restrained_decoding.py'
    the main simply read argument then read input file to setup the big loop iterating over all test prompt, program simply finish with creating the output file with all gathered data.

## Performance analysis
    after a lot of tweaking in promt and logits bias, i think i achieved a relativily correct accuracy and the only speed problem can ocur if a lot of parameter candidate are judged not good multiple times in a row which is pretty rare so the speed should not be a problem


## Challenges faced
    a lot of them but the biggest one was knowing when to finish generating text when a string parameter forces me to let the model generate on its own, to solve this problem i found not one but multiple solution that works well together.
    Solution :
    1.
    in the prompt i tell the model to finish with empty lines, if i detect that i end generation
    2.
    i end the prompt with a quote, if the model terminate the quote, i end generation
    3.
    if the llm create a <|im_end|> or <|end_of_text|> token i end generation

    with theses 3 end condition the llm is pretty good at telling when its work is done


## Testing strategy
    i continuously run the program with given test exercice and promt of my own and sometimes function of my own


## Example usage
    uv sync
    uv run python src [flag for file emplacement if necessary]