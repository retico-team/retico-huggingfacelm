import retico_core
from retico_core import abstract
from retico_core.text import SpeechRecognitionIU, TextIU

from retico_huggingfacelm.huggingface_lm_client import HuggingfaceLMClient


class HuggingfaceLMModule(abstract.AbstractModule):

    def __init__(self, lm_client : HuggingfaceLMClient):
        super().__init__()
        self.lm_client = lm_client

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

    def generate_model_output(self, last_commit_sentence):

        response = self.lm_client.generate_response(last_commit_sentence)
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
