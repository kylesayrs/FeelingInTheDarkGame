from typing import Dict, List

import copy
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from models.mood import MoodState
from models.adversary import Adversary
from config import GameConfig, default_config

class Game():
    def __init__(self):
        self.game_config = default_config
        self._chat_tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
        self._chat_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400M-distill")
        self._chat_model.to(self.game_config.device)

        self.chat_history = []

    def _get_chat_loss(self, user_input):
        input_text = "\n".join(self.chat_history[2:])
        input_tokens = self._chat_tokenizer(input_text, return_tensors="pt")
        input_tokens.to(self.game_config.device)

        label_tokens = self._chat_tokenizer(
            user_input,
            # don't return attention mask
            return_tensors="pt"
        )
        label_tokens.to(self.game_config.device)

        output = self._chat_model.forward(
            **input_tokens,
            labels=label_tokens["input_ids"]
            # I think there's an option to not return stuff other than the loss
        )

        return output.loss

    def run(self):
        adversary = Adversary("angry", self.game_config)

        target_words = []
        taboo_words = []
        target_words_left = copy.deepcopy(target_words)

        while True:
            user_input = input("User: ")

            chat_loss = self._get_chat_loss(user_input)
            if chat_loss > self.game_config.loss_threshold:
                adversary.increment_mood()

            for taboo_word in taboo_words:
                if taboo_word in user_input:
                    adversary.increment_hallucinations()

            self.chat_history.append(user_input)
            adversary_response = adversary.generate_response(self.chat_history)

            print(adversary_response)

            if target_words in adversary_response:
                target_words_left.remove(target_word)

            break

if __name__ == "__main__":
    game = Game()
    game.run()
