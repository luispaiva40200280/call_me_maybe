# from src.data_loader.loaders import parse_file
from .decoding import PromptSchema, FunctionDef
import argparse
import sys
from . import ft_time

"""
def run_pipeline(prompts: list[any], funcs: list[any]) -> None:
       try:
        prompts = parse_file(
            path=ARGS.input,
            type="prompt")
        functions_calling = parse_file(
            path=ARGS.functions_definition,
            type="definitions"
        )
        pipeline = Pipeline(model=Small_LLM_Model(ARGS.model),
                            func_call=functions_calling,
                            output_path=ARGS.output)
        begin = time.time()
        for i, prom in enumerate(prompts, start=1):
            print(f"testing Prompt pipeline nbr: {i}")
            print(f"Prompt: {prom.prompt}")
            start = time.time()
            gen = pipeline.generate(prompt=prom.prompt)
            pipeline.json_file_loader(prompt_text=prom.prompt,
                                        generated_str=gen)
            print("\n")
            end = time.time()
            time_structure = time.gmtime(end - start)
            formatted_string = time.strftime("%M:%S", time_structure)
            print(f"Time of excution: {formatted_string}")
            print("----" * 35)

        end = time.time()
        time_structure = time.gmtime(end - begin)
        string = time.strftime("%M:%S", time_structure)
        print(f"Total time of processing: {string}")
    except Exception as e:
        print(f"\033[31m Error: {e}\033[0m")
    except KeyboardInterrupt:
        sys.exit(0)
"""


def load_imports(ARGS: argparse.Namespace) -> tuple[
                    list[PromptSchema], list[FunctionDef]]:
    """
    """
    from .data_loader import load_prompt_data, load_func_data

    prompts = load_prompt_data(path=ARGS.input)
    func_calling = load_func_data(path=ARGS.functions_definition)

    return (prompts, func_calling)


@ft_time
def main() -> None:
    """
    Main function to create, use the llm model and the pipeline
    """
    from .data_loader import ArgsParser
    from llm_sdk import Small_LLM_Model
    from src.engine.fsm import StateMachine
    from src.engine.genarator import Generator
    from src.engine.pipeline import Pipeline

    # loading all data necessary to use the pipeline
    try:
        ARGS = ArgsParser()
        prompts, func_calling = load_imports(ARGS=ARGS)
    except ValueError as err:
        for index, e in enumerate(err.errors(), start=1):
            print(f"({index})\033[31m Error:\033[0m {e['msg']}\n")
        sys.exit(1)
    except (FileExistsError, FileNotFoundError) as e:
        print(f"\033[31m Error:\033[0m {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\033[32mKeybord Interropt with success\033[0m")
        sys.exit(0)

    # loading the pipeline and using it
    try:
        model = Small_LLM_Model(model_name=ARGS.model)
        fsm = StateMachine(schemas=func_calling)

        generator = Generator(
            model=model,
            fsm=fsm,
            func_call=func_calling,
        )
        pipeline = Pipeline(
            generator=generator,
            output_path=ARGS.output,
        )
        pipeline.run(prompts=[p.prompt for p in prompts])
    except Exception as e:
        print(f"error: {e}")
    except KeyboardInterrupt:
        print("\n\033[32mKeybord Interrupt with success\033[0m")
        sys.exit(0)


if __name__ == "__main__":
    main()
