import json
import os.path as osp
from typing import Union


class Prompter(object):
    __slots__ = ("template", "_verbose", "real_template")

    def __init__(self, template_name: str = "", verbose: bool = False, real_template=True):
        self._verbose = verbose
        self.real_template = real_template
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "short"

        if not real_template:
            self.template = None
            return
        file_name = osp.join("./prompt_templates", f"{template_name}.json")
        if not osp.exists(file_name):
            file_name = osp.join("../prompt_templates", f"{template_name}.json")
            if not osp.exists(file_name):
                raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
            self,
            input_data: str,
            output_label: str = None,
            instruction: str = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if instruction is not None:
            input_data = f"{instruction}\n{input_data}"

        if not self.real_template:
            if output_label is None:
                return input_data
            return f"{input_data}{output_label}"

        t = self.template["prompt"]
        res = t.format(
            input=input_data
        )
        if output_label:
            res = f"{res}{output_label}"
        return res

    def get_response(self, output: str) -> str:
        if not self.real_template:
            return output
        return output.split(self.template["response_split"])[1].strip()


if __name__ == "__main__":
    p = Prompter()
    p.generate_prompt("qwer")
