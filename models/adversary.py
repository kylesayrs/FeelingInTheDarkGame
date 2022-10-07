from typing import List, Any, Dict

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification
)

from models.mood import MoodState
from config import GameConfig

class Adversary():
    def __init__(
        self,
        mood_name: str,
        game_config: GameConfig,
    ):
        self._game_config = game_config
        self._mood_state = MoodState.from_game_config(mood_name, self._game_config)

        self._emotion_tokenizer = AutoTokenizer.from_pretrained(game_config.emotion_model_config.pretrained_model_name_or_path)
        self._emotion_model = AutoModelForSequenceClassification.from_pretrained(game_config.emotion_model_config.pretrained_model_name_or_path)
        self._emotion_model.to(self._game_config.device)

        self._chat_tokenizer = AutoTokenizer.from_pretrained(game_config.chat_model_config.pretrained_model_name_or_path)
        self._chat_model = AutoModelForSeq2SeqLM.from_pretrained(game_config.chat_model_config.pretrained_model_name_or_path)
        self._chat_model.to(self._game_config.device)

        bad_words_tokens = self._chat_tokenizer(
            " ".join(self._game_config.bad_words),
            # don't return attention mask
            # don't include special tokens
            add_special_tokens=False,
            #return_attention_mask=False,
            #return_special_tokens_mask=False,
            return_tensors="pt"
        )
        print(bad_words_tokens)
        self._bad_words_ids = bad_words_tokens["input_ids"].tolist()


    def _get_mood_logits(self, response):
        return self._mood_classifier(response)


    def increment_mood(self):
        print("increment_mood")
        self._mood_state.mood_priority = max(
            self._mood_state.mood_priority - 1,
            1
        )


    def increment_hallucinations(self):
        self._mood_state.hallucinations_priority = min(
            self._mood_state.hallucinations_priority + 1,
            len(self._mood_state.hallucinations) - 1
        )


    def generate_response(self, chat_history):
        print(chat_history)

        input_text = "\n".join(chat_history[-2:])
        print(input_text)
        input_tokens = self._chat_tokenizer(input_text, return_tensors="pt")
        input_tokens.to(self._game_config.device)
        print(input_tokens)

        hallucination_tokens = self._chat_tokenizer(
            self._mood_state.hallucinations[self._mood_state.hallucinations_index],
            # Don't return attention mask
            return_tensors="pt",
        )
        hallucination_tokens.to(self._game_config.device)

        responses = self._chat_model.generate(
            **input_tokens,
            **self._game_config.chat_model_config.generate_kwargs,
            decoder_input_ids=hallucination_tokens["input_ids"],
            bad_words_ids=self._bad_words_ids,
        )

        mood_logits = [_get_mood_logits(response) for response in responses]
        # processing to get scores
