"""
Pipeline: runs a batch of prompts through a Generator and persists the
results. Owns orchestration and file I/O only -- all decoding logic
lives in Generator (src/engine/generator.py).
"""
from typing import Any
from .genarator import Generator
from pydantic import BaseModel, ConfigDict
from pathlib import Path


class Pipeline(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    generator: Generator

    output_path: str

    def model_post_init(self, __context: Any) -> None:
        """
        """
        Path(self.output_path).unlink(missing_ok=True)

    def run(self, prompts: list[str]) -> None:
        """
        Runs every prompt through the generator, appending each result
        to output_path. A single prompt failing (bad JSON, grammar
        dead-end, exceeded max_tokens, unknown function/parameter) is
        logged and skipped rather than aborting the whole batch -- this
        is the gap that let one bad prompt crash an entire run before.
        """
        for prompt in prompts:
            try:
                result = self.generator.generate(prompt)
            except ValueError:
                result = {"name": "", "parameters": {}}
            self._append_result(prompt, result)

    def _append_result(self, prompt: str, result: dict) -> None:
        """Appends one {"prompt", "name", "parameters"} entry to the
        output JSON array, tolerating a missing or corrupted file."""
        import json
        file_path = Path(self.output_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            parsed_result = json.loads(result)
        except json.JSONDecodeError:
            parsed_result = {"name": "", "parameters": {}}

        entry = {
            "prompt": prompt,
            "name": parsed_result.get("name", ""),
            "parameters": parsed_result.get("parameters", {}),
        }

        data = []
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                print("Existing output file was corrupted; starting fresh")

        data.append(entry)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
