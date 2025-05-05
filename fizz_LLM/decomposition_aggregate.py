from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import prompts_example
import prompts_aggregate
import warnings
from nltk.tokenize import sent_tokenize
import re

warnings.filterwarnings('ignore')

model_map = {
    "orca2": {"model_ckpt": "microsoft/Orca-2-7b"},
    "mistral": {"model_ckpt": "mistralai/Mistral-7B-Instruct-v0.2"},
    "zephyr": {"model_ckpt": "HuggingFaceH4/zephyr-7b-beta"}
}

class DawonClass:
    def __init__(self, model_name="orca2", device="cuda"):
        assert model_name in model_map.keys(), f"Wrong model name: {model_name}"

        self.model_name = model_name
        self.model_ckpt = model_map[self.model_name]["model_ckpt"]
        self.device = device
        self.model = None
        self.tokenizer = None

    def load_lm(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_ckpt,
            use_fast=False,
            padding_side="left",
            #add_special_tokens=False
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_ckpt,
            torch_dtype=torch.float16
        ).to(self.device)

    def split_sentences(self, text):
        sentences = sent_tokenize(text)
        return [sent for sent in sentences if len(sent) > 10]

    def get_prompt(self, text, task="decompose", num_shot=1):
        if task == "decompose":
            if self.model_name == "orca2":
                return prompts_example.orca_format(text)
            elif self.model_name == "mistral":
                return prompts_example.mistral_format(text)
            elif self.model_name == "zephyr":
                return prompts_example.zephyr_format(text)
        elif task == "aggregate":
            if self.model_name == "orca2":
                return prompts_aggregate.orca_format(text, num_shot=num_shot)
            elif self.model_name == "mistral":
                return prompts_aggregate.mistral_format(text)
            elif self.model_name == "zephyr":
                return prompts_aggregate.zephyr_format(text)
        else:
            raise ValueError(f"Unknown task: {task}")

    def tokenize(self, prompt_text):
        if self.model_name == "orca2":
            model_inputs = self.tokenizer(prompt_text, return_tensors="pt")
            return model_inputs['input_ids']
        else:
            model_inputs = self.tokenizer.apply_chat_template(prompt_text, return_tensors="pt")
            return model_inputs

    def generate_not_batch(self, input_id):
        with torch.no_grad():
            return self.model.generate(
                input_id.to(self.device),
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

    def decode_not_batch(self, generated_id):
        return self.tokenizer.batch_decode(generated_id, skip_special_tokens=True)

    def generate_and_decode(self, prompt_text):
        input_ids = self.tokenize(prompt_text)
        generated_ids = self.generate_not_batch(input_ids)
        decoded = self.decode_not_batch(generated_ids)
        return decoded

    def preprocess_decoded_ids(self, decoded, task="decompose"):
        text = decoded[0]
        if self.model_name == 'orca2':
            if task == "decompose":
                text = text.split('<|im_start|> assistant')[-1]
            elif task == "aggregate":
                text = text.split('<|im_start|>assistant')[-1]
        elif self.model_name == 'zephyr':
            text = text.split('<|assistant|>')[-1]
        elif self.model_name == 'mistral':
            text = text.split('[/INST]')[-1]

        preprocessed_text = "".join(text.split('\n-')).replace('\n', '')
        preprocessed_text = preprocessed_text.replace("<|im_start|>", "")
        return preprocessed_text

    def atomic_facts_decompose(self, text):
        if self.model is None:
            self.load_lm()

        sentences = self.split_sentences(text)
        full_atomic_facts = ""

        for sentence in sentences:
            prompt = self.get_prompt(sentence, task="decompose")
            decoded = self.generate_and_decode(prompt)
            preprocessed = self.preprocess_decoded_ids(decoded, task="decompose")
            full_atomic_facts += preprocessed

        return full_atomic_facts

    def orca_format_with_truncation_handling(self, user_prompt, max_length=2048, min_shot=1, max_shot=10):
        for num_shot in range(max_shot, min_shot - 1, -1):
            prompt = self.get_prompt(user_prompt, task="aggregate", num_shot=num_shot)
            input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
            if input_ids.shape[1] <= max_length:
                return prompt
        return self.get_prompt(user_prompt, task="aggregate", num_shot=1)

    def aggregate(self, sum_texts, doc_texts, scores):
        if self.model is None:
            self.load_lm()

        decisions = []
        for summary, docs, scs in zip(sum_texts, doc_texts, scores):
            evidence_lines = ""
            for i, (doc, sc) in enumerate(zip(docs, scs), 1):
                evidence_lines += f"  [{i}] {doc}\n"
                evidence_lines += f"     Entailment: {sc[0]:.2f}, Contradiction: {sc[1]:.2f}, Neutral: {sc[2]:.2f}\n"

            user_prompt = f"Summary: {summary}\nDocument Sentences with NLI Scores:\n{evidence_lines}"

            full_prompt = self.orca_format_with_truncation_handling(user_prompt)
            input_ids = self.tokenizer(full_prompt, return_tensors="pt")["input_ids"]
            generated_ids = self.generate_not_batch(input_ids)
            decoded = self.decode_not_batch(generated_ids)

            try:
                if "<|im_start|>assistant" in decoded[0]:
                    response = decoded[0].split("<|im_start|>assistant")[-1]
                else:
                    response = decoded[0]

                response = response.strip().split("<|im_end|>")[0]
                match = re.search(r'\b[01]\b', response)
                answer = int(match.group()) if match else 0
            except:
                answer = 0

            decisions.append(answer)

        return decisions

# def main():
    # model = Dawonclass()

    # text = "lisa courtney, of hertfordshire, has spent most of her life collecting pokemon memorabilia."
    # atomic_facts = model.atomic_facts_decompose(text)
    # print("[Atomic Facts]")
    # print(atomic_facts)

    # # 예시용 (sum_texts, doc_texts, scores 데이터가 필요)
    # # decisions = model.aggregate([...], [...], [...])
    # # print(decisions)

# if __name__ == "__main__":
#     main()
