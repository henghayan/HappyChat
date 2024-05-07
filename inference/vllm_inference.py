import argparse
import gc
import json
import time

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

from vllm import LLM, SamplingParams


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

        generate_config = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "num_beams": num_beams,
            "max_new_tokens": max_new_tokens,
            "stop_token_ids": [tokenizer.eos_token_id]
        }

        if stream_output:
            def generate_with_callback(prompts, callback=None, **kwargs):
                with torch.no_grad():
                    sampling_params = SamplingParams(temperature=temperature,
                                                     top_p=top_p,
                                                     top_k=top_k,
                                                     stop_token_ids=kwargs.get("stop_token_ids", []),
                                                     max_tokens=max_new_tokens
                                                     )
                    # client.generate(prompts, sampling_params=sampling_params)
                    engine = client.llm_engine
                    engine.add_request(time.time(), prompts, sampling_params)
                    while engine.has_unfinished_requests():
                        step_outputs = engine.step()
                        for output in step_outputs:
                            if not output.finished:
                                callback(output)
                            else:
                                break

            def generate_with_streaming(prompts, **kwargs):
                print("start init generate_with_streaming")
                return Iteratorize(
                    generate_with_callback, prompts, kwargs, callback=None
                )

            with generate_with_streaming(prompt, **generate_config) as generator:
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
    client = LLM(model=model_path, quantization="AWQ", tensor_parallel_size=2)
    # client = LLM(model=model_path, quantization="AWQ")

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
        # self.res = ""
        self.break_sign = False

        def _callback(response):
            if self.stop_now:
                raise ValueError
            if not response.finished:
                self.q.put(response.outputs[0].text)
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
            return obj

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
    # print(transformers.__version__)55
    # main("/data2/awq_llm3_8", "/data2/awq_llm3_8", "cuda", False, get_args().c_8bit, get_args().lora)
    # test_gr()

    main("/data/awq_llm3_70", "/data/awq_llm3_70", "cuda", False, get_args().c_8bit, get_args().lora)

    # test_data = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    # Always answer with Haiku<|eot_id|><|start_header_id|>user<|end_header_id|>
    # I am going to Paris, what should I see?<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''
    #
    # test_2 = "Hello, my name is"
    #
    # sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    # llm = LLM(model="/data/awq_llm3_8", quantization="AWQ")
    # outputs = llm.generate([test_data, test_2], sampling_params)
    # print("output", outputs)