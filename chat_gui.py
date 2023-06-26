import argparse
import gc

import gradio as gr
import torch
import transformers
from transformers import GenerationConfig
import traceback
from queue import Queue
from threading import Thread

from model_loader import load_model, compress_8bit
from utils.prompter import Prompter


def wrap_evaluate(model, tokenizer, device, prompt_template=""):
    prompter = Prompter(prompt_template)

    def evaluate(
            input_data,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            max_new_tokens=128,
            stream_output=False,
            **kwargs,
    ):
        prompt = prompter.generate_prompt(input_data)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }

        if stream_output:
            def generate_with_callback(callback=None, **kwargs):
                kwargs.setdefault(
                    "stopping_criteria", transformers.StoppingCriteriaList()
                )
                kwargs["stopping_criteria"].append(
                    Stream(callback_func=callback)
                )
                with torch.no_grad():
                    model.generate(**kwargs)

            def generate_with_streaming(**kwargs):
                print("start init generate_with_streaming")
                return Iteratorize(
                    generate_with_callback, kwargs, callback=None
                )

            with generate_with_streaming(**generate_params) as generator:
                for output in generator:
                    # new_tokens = len(output) - len(input_ids[0])
                    decoded_output = tokenizer.decode(output)
                    print(decoded_output)
                    if output[-1] in [tokenizer.eos_token_id]:
                        break

                    yield prompter.get_response(decoded_output)
            return  # early return for stream_output

        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        yield prompter.get_response(output)

    return evaluate


def main(path, tokenizer_path, device="cuda:0", share=False, load_8bit=False, lora=False):
    print("model_path", tokenizer_path)
    print("token loading ...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, use_fast=False)
    print("tokenizer loaded ok")
    print("start load model...")
    model = load_model(path)

    if load_8bit:
        compress_8bit(model, device)
    # if lora:
    #     lora_model(model)

    # model.to(device)
    # model.eval()
    # gc.collect()
    # torch.cuda.empty_cache()
    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)
    print("start init evaluate_func ")
    #
    GUI(model, tokenizer, device, share=share)


def GUI(model, tokenizer, device, share=False):
    gc.collect()
    torch.cuda.empty_cache()
    evaluate_func = wrap_evaluate(model, tokenizer, device)
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
                minimum=1, maximum=4, step=1, value=4, label="Beams"
            ),
            gr.components.Slider(
                minimum=1, maximum=2000, step=1, value=128, label="Max tokens"
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
        description="测试",
    ).queue().launch(server_name="0.0.0.0", share=share)


class Stream(transformers.StoppingCriteria):
    def __init__(self, callback_func=None):
        self.callback_func = callback_func

    def __call__(self, input_ids, scores) -> bool:
        if self.callback_func is not None:
            self.callback_func(input_ids[0])
        return False


class Iteratorize:

    def __init__(self, func, kwargs={}, callback=None):
        self.mfunc = func
        self.c_callback = callback
        self.q = Queue()
        self.sentinel = object()
        self.kwargs = kwargs
        self.stop_now = False

        def _callback(val):
            if self.stop_now:
                raise ValueError
            self.q.put(val)

        def gentask():
            try:
                ret = self.mfunc(callback=_callback, **self.kwargs)
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
                minimum=1, maximum=4, step=1, value=4, label="Beams"
            ),
            gr.components.Slider(
                minimum=1, maximum=2000, step=1, value=128, label="Max tokens"
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
    main(get_args().model_path, get_args().model_path, "cuda", False, get_args().c_8bit, get_args().lora)
    # test_gr()
