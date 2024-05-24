import json

from app.model_api import ModelAPI


class ParallelHandler:
    def __init__(self):
        self.model_api = ModelAPI()
        self.org_context_map = {}

    def _split_context_sliding_window(self, contexts, max_length=512, window_size=128):
        chunks = []
        init_index = 0
        for org_ctx_index in range(len(contexts)):
            context = contexts[org_ctx_index]
            while len(context) > max_length:
                chunk, context = context[:max_length - window_size], context[max_length - window_size:]
                chunks.append(chunk)
                self.org_context_map[init_index] = org_ctx_index
                init_index += 1

            chunks.append(context)
            self.org_context_map[init_index] = org_ctx_index
            init_index += 1

        return chunks

    def get_answers(self, questions, contexts):
        # 超长的contexts做窗口滑动切割，做模型并行处理
        contexts = self._split_context_sliding_window(contexts)

        formatted_prompts = self.model_api.format_prompt(questions, contexts)

        # 调用大模型API
        raw_outputs = self.model_api.call_model_api(formatted_prompts)

        # 处理和去重API结果
        answers_dict = {}
        for output_index in range(len(raw_outputs)):
            output = raw_outputs[output_index]
            data_source_index = self.org_context_map[output_index]
            try:
                output = self.get_json_from_model_api(output)
            except Exception as e:
                raise Exception("model api data not json invalid: {}".format(output))

            for question in questions:
                if answers_dict.get(question, None) is None:
                    answers_dict[question] = []

                answer = output.get(question, None)
                if answer is None or answer == "":
                    continue

                # 对数据做冗余储存，保留所有数据源线索
                answers_dict[question].append({"value": answer, "data_source_index": data_source_index})

        # # 去重每个问题的答案
        # for question in answers_dict:
        #     answers_dict[question] = list(set(answers_dict[question]))

        return answers_dict

    @staticmethod
    def get_json_from_model_api(output_str):
        output_str = output_str.strip()

        return json.loads(output_str)
