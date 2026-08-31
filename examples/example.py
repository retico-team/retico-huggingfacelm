from retico_huggingfacelm.huggingface_lm_module import HuggingfaceLMModule
from retico_huggingfacelm.huggingface_lm_client import HuggingfaceLMClient

print("Starting hugging face lm...",end="")
# device = "cuda:0"
device = "cpu"

#### Make a client.
#instantiate an llm client from Huggingface with desired model and defaults
#this shortcut uses default tokenizer, streamer, and model paramters, including
#a text streamer that streams output to stdout. use the constructor to avoid this.
# - note that llm_cclient has other parameters such as maximum output length, etc, you should explore
llm_client = HuggingfaceLMClient.quick_from_checkpoint(
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    device=device,
    temperature=0.7
)  # very small, quick, good for small inferences. not a healthy chatbot. Funny.



#create a retico llm module using this client.
llm_module = HuggingfaceLMModule(llm_client)

#Now you have a retico module that can be connected to a retico network.
#-- this module takes in SpeechRecognition UIs and generates TextIUs
#### example network (not instantiated above) and how this may fit
# into the network
# microphone_module.subscribe(asr_module)
# asr_module.subscribe(llm_module)
# llm_module.subscribe(tts_module)
# tts_module.subscribe(speaker_module)

#without we instead use the llm_client directly to generate an output.
llm_client.set_system_role("You are a grumpy Wizard who does not talk a lot.")
response = llm_client.generate_response("Who are you?")
print("done")  # we don't print the response since the default streamer dumps to stdout
