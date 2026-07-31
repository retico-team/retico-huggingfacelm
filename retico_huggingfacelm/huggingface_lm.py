import os
import retico_core
import sys
import torch
from retico_core import abstract
from retico_core.text import SpeechRecognitionIU, TextIU
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, TextIteratorStreamer


class HuggingfaceLM(abstract.AbstractModule):

    @classmethod
    def from_checkpoint(cls, checkpoint, device="cpu"):
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True).to(device)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        return cls(device, tokenizer, model, streamer)

    def __init__(self, device, tokenizer, model, streamer):
        super().__init__()
        self._device = device
        self._tokenizer = tokenizer
        self._model = model
        self._streamer = streamer
        self._system_role = "You are a friendly chatbot who responds to questions"  # just a default.

    @staticmethod
    def name():
        return "Hugging Face LM Module"

    @staticmethod
    def description():
        return "A module running Hugging Face language model for real-time dialogue."

    @staticmethod
    def input_ius():
        return SpeechRecognitionIU

    @staticmethod
    def output_iu():
        return TextIU

    def process_update(self, update_message):
        send_prompt = False

        for iu, ut in update_message:

            if ut == abstract.UpdateType.ADD:
                self.current_output.append(iu)
            elif ut == abstract.UpdateType.REVOKE:
                self.revoke(iu)
            elif ut == abstract.UpdateType.COMMIT:
                send_prompt = True

        if send_prompt:
            send_prompt = False
            last_commit_sentence = ""
            for unit in self.current_output:
                last_commit_sentence += f"{unit.text} "
            self.current_output = []

            if len(last_commit_sentence) > 0:
                # print('user:', last_commit_sentence)
                self.generate_model_output(last_commit_sentence)

    def set_system_role(self, role):
        self._system_role = role

    def generate_model_output(self, last_commit_sentence):

        messages = [
            {"role": "system",
             "content": self._system_role},
            {"role": "user",
             "content": last_commit_sentence},
        ]

        tokenized_chat = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        if isinstance(tokenized_chat, torch.Tensor):  # sometimes we get a tensor, other times a dictionary
            input_ids = tokenized_chat.to(self._device)

        else:
            input_ids = tokenized_chat["input_ids"].to(self._device)

        input_length = input_ids.shape[1]

        with torch.no_grad():
            output_tokens = self._model.generate(
                input_ids,
                max_new_tokens=500,
                temperature=0.2,
                top_p=0.9,
                do_sample=True,
                streamer=self._streamer
            )

        response = self._tokenizer.decode(output_tokens[0][input_length:], skip_special_tokens=True)
        words = response.split()

        current_iu = None
        for word in words:
            current_iu = self.create_iu()
            current_iu.payload = word
            update_message = retico_core.UpdateMessage.from_iu(current_iu, retico_core.UpdateType.ADD)
            self.append(update_message)

        # Send singular COMMIT to signal end of output/response for a given prompt
        if current_iu is not None:
            update_message = retico_core.UpdateMessage.from_iu(current_iu, retico_core.UpdateType.COMMIT)
            self.append(update_message)

    def process_revoke(self, iu):
        pass
