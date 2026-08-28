import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, TextIteratorStreamer

class HuggingfaceLMClient:

    @classmethod
    def quick_from_checkpoint(cls,
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

    ########## API interface to the LM
    # - set the system role, defines the instructions to the LM (the system role)
    # - set max history turns, defines how many exchanges to keep in the prompt to the LM
    # - clear history, as described, keeps the role
    # - generate response, takes a next query, compiles the role and history, and returns the result.

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


    def generate_response(self, query):

        self._message_history.append( {"role": "user", "content": query}   )

        tokenized_chat = self._tokenizer.apply_chat_template(
            self._message_history,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        if isinstance(tokenized_chat, torch.Tensor):  # sometimes we get a tensor, other times a dictionary, based on LM version
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
        self._add_response_and_truncate(response)
        return response