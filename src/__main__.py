from llm_sdk import Small_LLM_Model
import argparse
from src.loaders import parse_file
from src.pipeline import Pipeline
from src import ft_time


def ArgsParser() -> argparse.Namespace:
    """
        Parser for the arguments needed to use the LLM

        RETURNS
            args (argparse.Namespace): arguments for the lllm function calling
    """
    parser = argparse.ArgumentParser(description="LLM Function Calling Tool")
    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to the functions definition JSON file",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to the input tests JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calling_results.json",
        help="Path to save the output JSON results",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B",
        help="Name of the language model to use"
    )
    args = parser.parse_args()
    return args


@ft_time
def main() -> None:
    """
    Main function to create and use the llm model and the pipeline
    """
    import time
    import sys
    start = time.time()
    ARGS = ArgsParser()
    try:
        prompts = parse_file(
            path=ARGS.input,
            type="prompt")
        functions_calling = parse_file(
            path=ARGS.functions_definition,
            type="definitions"
        )
        pipeline = Pipeline(model=Small_LLM_Model(ARGS.model),
                            func_call=functions_calling)
        for i, prom in enumerate(prompts, start=1):
            print(f"testing Prompt pipeline nbr: {i}")
            print(f"Prompt: {prom.prompt}")
            start = time.time()
            pipeline.generate(prompt=prom.prompt)
            print("\n")
            end = time.time()
            time_structure = time.gmtime(end - start)
            formatted_string = time.strftime("%M:%S", time_structure)
            print(f"Time of excution: {formatted_string}")
            print("----" * 35)
    except Exception as e:
        print(f"\033[31m Error: {e}\033[0m")
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
