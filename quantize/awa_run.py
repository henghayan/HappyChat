from vllm import LLM, SamplingParams

sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=256)
llm = LLM(
    model="/data2/llm3-8",
    quantization='awq',
    dtype='half',
    gpu_memory_utilization=.95,
    max_model_len=4096
)
output = llm.generate(['test'], sampling_params)
print(output[0].outputs[0].text)


