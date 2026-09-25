from llm_sdk import Small_LLM_Model
import argparse
from src.loaders import FileHandler
from src.pipeline import Pipeline


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


def main() -> None:
    """
    Main function to create and use the llm model and the pipeline
    """
    ARGS = ArgsParser()
    try:
        prompts = FileHandler(
            path_name=ARGS.input,
            file_type="prompt").parse_file()
        functions_calling = FileHandler(
            path_name=ARGS.functions_definition,
            file_type="definitions"
        ).parse_file()
        pipeline = Pipeline(model=Small_LLM_Model(ARGS.model),
                            func_call=functions_calling)
        print(pipeline)
        print(functions_calling)
        print(prompts)
    except Exception as e:
        print(f"\033[31m Error: {e}\033[0m")


if __name__ == "__main__":
    main()
