import argparse
import gc
import json

import gradio as gr
import torch
import transformers
from transformers import GenerationConfig, AwqConfig
# from transformers import QuantoConfig
import traceback
from queue import Queue
from threading import Thread

from utils.prompter import Prompter
import transformers

import mii


def wrap_evaluate(client, tokenizer, device, prompt_template=""):
    prompter = Prompter(prompt_template, real_template=False)

    def evaluate(
            input_data,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=1,
            max_new_tokens=128,
            stream_output=False,
            **kwargs,
    ):
        # prompt = prompter.generate_prompt(input_data)
        prompt = json.loads(input_data)
        prompt = tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True
        )
        print("prompt", prompt)

        generate_params = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_new_tokens": max_new_tokens,
        }

        if stream_output:
            def generate_with_callback(prompts, callback=None, **kwargs):
                with torch.no_grad():
                    client.generate(prompts, streaming_fn=callback, **kwargs)

            def generate_with_streaming(prompts, **kwargs):
                print("start init generate_with_streaming")
                return Iteratorize(
                    generate_with_callback, prompts, kwargs, callback=None
                )

            with generate_with_streaming(prompt, **generate_params) as generator:
                for output in generator:
                    item_res = prompter.get_response(output)
                    if item_res == "<break>":
                        break
                    yield item_res
            return  # early return for stream_output

        # Without streaming
        with torch.no_grad():
            generation_output = client.generate(
                prompts=prompt,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        yield prompter.get_response(output)

    return evaluate


def main(model_path, tokenizer_path, device="cuda:0", share=False, load_8bit=False, lora=False):
    print("model_path", tokenizer_path)
    print("token loading ...")
    # tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, use_fast=False)
    print("tokenizer loaded ok")
    print("start load model...")
    # max_memory_mapping = {"cpu": "32GB", 0: "4GB", 1: "4GB", 2: "4GB"}
    client = mii.serve(model_path, tensor_parallel=2)

    print("start init evaluate_func ")
    #
    GUI(client, tokenizer, device, share=share)


def GUI(client, tokenizer, device, share=False):
    gc.collect()
    torch.cuda.empty_cache()
    evaluate_func = wrap_evaluate(client, tokenizer, device)
    print("start init gui")
    gr.Interface(
        fn=evaluate_func,
        inputs=[
            gr.components.Textbox(lines=2, label="Input", placeholder="none"),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.1, label="Temperature"
            ),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.75, label="Top p"
            ),
            gr.components.Slider(
                minimum=0, maximum=100, step=1, value=40, label="Top k"
            ),
            gr.components.Slider(
                minimum=1, maximum=4, step=1, value=1, label="Beams"
            ),
            gr.components.Slider(
                minimum=1, maximum=8000, step=1, value=128, label="Max tokens"
            ),
            gr.components.Checkbox(label="Stream output"),
        ],
        outputs=[
            gr.inputs.Textbox(
                lines=5,
                label="Output",
            )
        ],
        title="呀哈哈",
        description="测试",
    ).queue().launch(server_name="0.0.0.0", share=share)


class Iteratorize:

    def __init__(self, func, prompts, kwargs={}, callback=None):
        self.mfunc = func
        self.c_callback = callback
        self.q = Queue()
        self.sentinel = object()
        self.kwargs = kwargs
        self.stop_now = False
        self.start_prompts = prompts
        self.res = ""
        self.break_sign = False

        def _callback(response):
            if self.stop_now:
                raise ValueError
            current_res = response[0]
            if current_res.finish_reason == "none":
                self.q.put(response[0].generated_text)
            else:
                self.break_sign = True

        def gentask():
            try:
                ret = self.mfunc(prompts=self.start_prompts, callback=_callback, **self.kwargs)
            except ValueError:
                pass
            except:
                traceback.print_exc()
                pass

            self.q.put(self.sentinel)
            if self.c_callback:
                self.c_callback(ret)

        self.thread = Thread(target=gentask)
        self.thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        obj = self.q.get(True, None)
        if obj is self.sentinel:
            raise StopIteration
        else:
            if self.break_sign:
                return "<break>"
            self.res += obj
            return self.res

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_now = True


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-model_path", type=str, default="")
    parser.add_argument("-c_8bit", type=bool, default=False)
    parser.add_argument("-lora", type=bool, default=False)
    return parser.parse_args()


def test_gr():
    def tmp(a, b, c, d, e, f, g):
        print(a, b, c, d, e, f, g)
        return

    gr.Interface(
        fn=tmp,
        inputs=[
            gr.components.Textbox(lines=2, label="Input", placeholder="none"),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.1, label="Temperature"
            ),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.75, label="Top p"
            ),
            gr.components.Slider(
                minimum=0, maximum=100, step=1, value=40, label="Top k"
            ),
            gr.components.Slider(
                minimum=1, maximum=4, step=1, value=1, label="Beams"
            ),
            gr.components.Slider(
                minimum=1, maximum=8000, step=1, value=128, label="Max tokens"
            ),
            gr.components.Checkbox(label="Stream output"),
        ],
        outputs=[
            gr.inputs.Textbox(
                lines=5,
                label="Output",
            )
        ],
        title="模型测试",
        description="直接在输入框内输入问题，暂时不支持历史记录上下文",
    ).queue().launch(server_name="0.0.0.0", share=False)


if __name__ == "__main__":
    print(transformers.__version__)
    main("/data2/awq_llm3_8", "/data2/awq_llm3_8", "cuda", False, get_args().c_8bit, get_args().lora)
    # test_gr()

    # main("/data2/llm3-70", "/data2/llm3-70", "cuda", False, get_args().c_8bit, get_args().lora)
