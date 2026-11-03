"""
"""
import argparse


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
