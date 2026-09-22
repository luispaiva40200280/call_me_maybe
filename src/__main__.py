from llm_sdk import Small_LLM_Model
import argparse
from src.loaders import FileHandler, load_vocab_lut


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
    import json   # use to test the loaders
    ARGS = ArgsParser()
    MODEL = Small_LLM_Model(ARGS.model)
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
        vocab = load_vocab_lut(MODEL)
        # TESTING IF THE FILES ARE OPENING AND STORING THE DATA CORRECTLY
        print("=== Prompts in json file")
        for prom in prompts:
            print(f"- {prom.values()}")

        print("\n=== Func Calling in json file ===")
        # Loop through the functions and print them cleanly
        for func in functions_calling:
            # .model_dump() (or .dict() in older Pydantic versions) converts
            # the model back to a readable dictionary
            print(json.dumps(func, indent=2))
        for index, item in enumerate(vocab.items(), start=1):
            print(f"{index} -- {item}")
        print(MODEL._model)
    except Exception as e:
        print(f"\033[31m Error: {e}\033[0m")


if __name__ == "__main__":
    main()
