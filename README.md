# Retico Hugging Face Module

A ReTiCo module that works with HuggingFace text generation models. The microphone captures user's speech, which passes through the ASR which is given as input to the Language Model. 
The output of the Language Model is generated and printed.

### Installation and requirements ###

* Clone retico_core:
https://github.com/retico-team/retico-core.git

* Clone retico_whisperasr:
https://github.com/retico-team/retico-whisperasr.git

* pip install pyaudio
* pip install pydub
* pip install webrtcvad

For HuggingFace login use this command and provide your token.

* huggingface-cli login

Some Hugging Face models will require authorization.
Go to the model's page and request access to the model.

Currently tested models: HuggingFaceTB/SmolLM2-135M-Instruct, meta-llama/Llama-3.2-3B-Instruct

### Example ###

```python
import os
import sys

os.environ['RETICO'] = "path/to/retico-core"
os.environ['WASR'] = "path/to/retico-whisperasr"

sys.path.append(os.environ['RETICO'])
sys.path.append(os.environ['WASR'])

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, TextIteratorStreamer
from retico_core.debug import DebugModule
from retico_core.audio import MicrophoneModule
from retico_whisperasr.whisperasr import WhisperASRModule
from retico_huggingfacelm.huggingface_lm import HuggingfaceLM

device = "cuda" if torch.cuda.is_available() else "cpu"

""" HuggingFace Model, Tokenzier, Model """
checkpoint = "HuggingFaceTB/SmolLM2-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(checkpoint,  trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(checkpoint,  trust_remote_code=True).to(device)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

mic = MicrophoneModule()
asr = WhisperASRModule(language='english')
debug = DebugModule(print_payload_only=True)
lm = HuggingfaceLM(device, tokenizer, model, streamer)

mic.subscribe(asr)
asr.subscribe(debug)
asr.subscribe(lm)

mic.run()
asr.run()
debug.run()
print(f"Hugging Face Model: {checkpoint}")
lm.run()

input()

mic.stop()
asr.stop()
lm.stop()
debug.stop()

```
