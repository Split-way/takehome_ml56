import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from abc import ABC, abstractmethod


class GPT(ABC):
    @abstractmethod
    def return_response(self, message: str) -> str:
        pass


class GPT2(GPT):
    model_name = "gpt2"

    def __init__(self):
        self.model = GPT2LMHeadModel.from_pretrained(GPT2.model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(GPT2.model_name)

    def _get_input_ids(self, message: str) -> torch.Tensor:
        input_ids = self.tokenizer.encode(message, return_tensors="pt")
        return input_ids

    def _get_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        attention_mask = torch.tensor([[1] * len(input_ids[0])])
        return attention_mask

    def _generate_output(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        temperature: int = 0.6,
        no_repeat_ngram_size: int = 2,
        max_new_tokens: int = 150,
    ) -> torch.Tensor:
        model_output = self.model.generate(
            input_ids=input_ids,
            do_sample=True,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        return model_output

    def _decode_output(self, model_output: torch.Tensor) -> str:
        response_message = self.tokenizer.decode(
            model_output[0], skip_special_tokens=True
        )
        return response_message

    def return_response(self, message: str) -> str:
        input_ids = self._get_input_ids(message)
        attention_mask = self._get_attention_mask(input_ids)
        output = self._generate_output(input_ids, attention_mask)
        response = self._decode_output(output)
        return response
