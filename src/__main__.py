from typing import Any
import argparse
from src.schemas import PromptInput
import json
from pathlib import Path


class FileHandler:
    """Class for opening and reading JSON files safely."""
    def __init__(self, path_name: str, file_type: str) -> None:
        self.path_name = path_name
        self.file_type = file_type

    def validate_data(self, data: list[dict[str, Any]]) -> None:
        """Validate of the data using the pydantic modules"""
        if self.file_type == "prompt":
            [PromptInput(**prompt) for prompt in data]
        # if the file is func calling needs another validation

    def parse_file(self) -> list[dict[str, Any]]:
        """
        Safely loads the JSON file.
        Note: This is now an instance method using `self`, not a classmethod.
        """

        path_file = Path(self.path_name)
        if not path_file.exists():
            raise FileNotFoundError(f"The {self.file_type} file was not \
found at: {self.path_name}")
        with open(path_file, 'r', encoding="utf-8") as file:
            data = json.load(file)
        self.validate_data(data)
        return data


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
    """ Main function to create and use the llm model and the pipeline"""
    ARGS = ArgsParser()
    print(ARGS)
    # model creation
    # model = ModelCreate(arguments.model)
    try:
        prompts = FileHandler(
            path_name=ARGS.input,
            file_type="prompt").parse_file()
        functions_calling = FileHandler(
            path_name=ARGS.functions_definition,
            file_type="definitions"
        ).parse_file()
        print("=== Prompts in json file")
        for prom in prompts:
            print(f"- {prom.values()}")

        print("\n=== Func Calling in json file ===")
        # Loop through the functions and print them cleanly
        for func in functions_calling:
            # .model_dump() (or .dict() in older Pydantic versions) converts
            # the model back to a readable dictionary
            print(json.dumps(func, indent=2))
    except Exception as e:
        print(f"\033[31m Error: {e}\033[0m")


if __name__ == "__main__":
    main()
