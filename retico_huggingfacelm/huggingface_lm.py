import os
import retico_core
import sys
import torch
from retico_core import abstract
from retico_core.text import SpeechRecognitionIU, TextIU
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, TextIteratorStreamer
from pprint import pformat


class HuggingfaceLM(abstract.AbstractModule):

    @classmethod
    def from_checkpoint(cls,
                        checkpoint,
                        device="cpu",
                        system_role="You are a friendly chatbot who responds to questions",  # just a default.
                        max_history_turns=10,  # how many exchange turns (q then a) to keep and feed to the model.
                        max_new_tokens=75,  # how many tokens (~words) it can produce at max
                        temperature=0.2  # between 0 and 1, higher means more randomness
                        ):
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True).to(device)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        return cls(device, tokenizer, model, streamer, system_role, max_history_turns, max_new_tokens, temperature)

    def __init__(self,
                 device,
                 tokenizer,
                 model,
                 streamer,
                 system_role,
                 max_history_turns,
                 max_new_tokens,
                 temperature
                 ):
        super().__init__()
        self._device = device
        self._tokenizer = tokenizer
        self._model = model
        self._streamer = streamer
        self._max_history_turns = max_history_turns
        self._message_history = []
        self._temperature = temperature
        self._max_new_tokens = max_new_tokens
        self.turns = 0
        self.set_system_role(system_role)

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

    # set the system role to be sent into the LLM from this point onward. Does not change history.
    def set_system_role(self, role):
        if (len(self._message_history) == 0):
            self._message_history.append({"role": "system", "content": role})
        else:
            self._message_history[0] = {"role": "system", "content": role}

    # set maximum number of exchange turns (prompt then response) to keep in history and feed to the model
    # If we have history allready and the newe limit is > existing history, truncates
    def set_max_history_turns(self, turns):
        self._max_history_turns = turns
        if turns < self.turns and turns > 0:  # truncate, keep the newest n turns
            self._message_history = [ self._message_history[0] ] + self._messages_history[-(turns * 2): ]

    def clear_history(self):
        system_role = self._message_history[0]["content"]
        self._message_history = []
        self.set_system_role(system_role)

    def _add_response_and_truncate(self, response):
        self._message_history.append({"role": "assistant", "content": response})
        self.turns += 1

        if self.turns > self._max_history_turns:
            self.turns -= 1
            self._message_history = [self._message_history[0]] + self._message_history[-(self.turns * 2):]

    def generate_model_output(self, last_commit_sentence):

        self._message_history.append( {"role": "user", "content": last_commit_sentence}   )

        tokenized_chat = self._tokenizer.apply_chat_template(
            self._message_history,
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
                max_new_tokens=self._max_new_tokens,
                temperature=self._temperature,
                top_p=0.9,
                do_sample=True,
                streamer=self._streamer
            )

        response = self._tokenizer.decode(output_tokens[0][input_length:], skip_special_tokens=True)

        # self._message_history.append({"role": "assistant", "content": response})
        # self.turns += 1
        self._add_response_and_truncate(response)
        print ("turns: "+str(self.turns)+" history  "+ pformat(self._message_history))

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
