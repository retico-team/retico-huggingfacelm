# retico-huggingfacelm
A ReTiCo module that works with Hugging Face text generation models. It takes the user's speech recognized by an ASR module, sends it to a language model, and outputs the model's response word by word.

## Installation and Requirements

To use the Hugging Face LM module you first need to install the retico-huggingfacelm package:
* Install retico-huggingfacelm:
```pip install retico-huggingfacelm```

  or, for the latest development version from GitHub:
```pip install git+https://github.com/retico-team/retico-huggingfacelm.git```

### Hugging Face login
Some Hugging Face models require authorization. Go to the model's page and request access, then log in with your token:
```huggingface-cli login```

## Modules

### `HuggingfaceLMClient`
A wrapper around a Hugging Face chat model that keeps the conversation history. Create one with `HuggingfaceLMClient.quick_from_checkpoint()`, which loads the model, tokenizer and a streamer that prints the response as it's generated, or with the constructor to supply your own.

#### Arguments (`quick_from_checkpoint`):
* `checkpoint` (str): the Hugging Face model to load, e.g. `"HuggingFaceTB/SmolLM2-135M-Instruct"`
* `device` (str): the device to run the model on, e.g. `"cpu"` or `"cuda:0"`; defaults to `"cpu"`
* `system_role` (str): the system instructions for the model; defaults to `"You are a friendly chatbot who responds to questions"`
* `max_history_turns` (int): the number of previous exchanges kept in the prompt; defaults to `10`
* `max_new_tokens` (int): the maximum number of tokens in a response; defaults to `75`
* `temperature` (float): between `0` and `1`; higher means more random responses; defaults to `0.2`

The client can also be used on its own with `generate_response(query)`, `set_system_role(role)`, `set_max_history_turns(turns)` and `clear_history()`.

### `HuggingfaceLMModule`
Subscribes to `SpeechRecognitionIU` and outputs `TextIU`. Once the user's speech is committed, it sends the text to the client and outputs the response one word per IU, followed by a COMMIT at the end of the response.

#### Arguments:
* `lm_client` (`HuggingfaceLMClient`): the client to generate responses with

### Example
**Note:** The example uses `MicrophoneModule` and `DebugModule` from `retico_core` and `WhisperASRModule` from [retico-whisperasr](https://github.com/retico-team/retico-whisperasr) (`pip install retico-whisperasr`).

```python
from retico_core.debug import DebugModule
from retico_core.audio import MicrophoneModule
from retico_whisperasr import WhisperASRModule
from retico_huggingfacelm import HuggingfaceLMClient, HuggingfaceLMModule

checkpoint = "HuggingFaceTB/SmolLM2-135M-Instruct"
llm_client = HuggingfaceLMClient.quick_from_checkpoint(checkpoint, device="cpu", temperature=0.7)

mic = MicrophoneModule()
asr = WhisperASRModule()
lm = HuggingfaceLMModule(llm_client)
debug = DebugModule(print_payload_only=True)

mic.subscribe(asr)
asr.subscribe(lm)
lm.subscribe(debug)

mic.run()
asr.run()
lm.run()
debug.run()

print(f"Hugging Face Model: {checkpoint}")
input()

mic.stop()
asr.stop()
lm.stop()
debug.stop()
```

The client can also be used without a network, e.g. to test a model:
```python
from retico_huggingfacelm import HuggingfaceLMClient

llm_client = HuggingfaceLMClient.quick_from_checkpoint("HuggingFaceTB/SmolLM2-135M-Instruct")
llm_client.set_system_role("You are a grumpy Wizard who does not talk a lot.")
llm_client.generate_response("Who are you?")  # the default streamer prints the response
```
