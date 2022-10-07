from typing import List, Any, Dict

import torch
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
        self._emotion_model.eval()

        self._chat_tokenizer = AutoTokenizer.from_pretrained(game_config.chat_model_config.pretrained_model_name_or_path)
        self._chat_model = AutoModelForSeq2SeqLM.from_pretrained(game_config.chat_model_config.pretrained_model_name_or_path)
        self._chat_model.to(self._game_config.device)
        self._chat_model.eval()

        bad_words_tokens = self._chat_tokenizer(
            " ".join(self._game_config.bad_words),
            add_special_tokens=False,
            return_attention_mask=False,
            return_special_tokens_mask=False,
            return_tensors="pt"
        )
        self._bad_words_ids = bad_words_tokens["input_ids"].tolist()


    def increment_mood(self):
        self._mood_state.mood_priority = max(
            self._mood_state.mood_priority - 1,
            1
        )


    def increment_hallucinations(self):
        self._mood_state.hallucinations_index = min(
            self._mood_state.hallucinations_index + 1,
            len(self._mood_state.hallucinations) - 1
        )


    def _get_mood_scores(self, response_texts):
        tokens = self._emotion_tokenizer(
            response_texts,
            padding=True,
            return_tensors="pt"
        )
        tokens.to(self._game_config.device)

        with torch.no_grad():
            emotion_outputs = self._emotion_model(**tokens)

        emotion_logits = emotion_outputs.logits
        scores = torch.nn.functional.softmax(emotion_logits[:, self._mood_state.mood_index], dim=0)

        return scores


    def _post_process_chat_outputs(self, outputs_with_hallucination, decoder_input_ids):
        outputs = outputs_with_hallucination[:, len(decoder_input_ids[0]):]
        response_texts = self._chat_tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True
        )
        response_texts = [
            ".".join(response_text.split(".")[:-1]) + "."
            for response_text in response_texts
        ]

        return response_texts


    def generate_response(self, chat_history):
        input_text = "\n".join(chat_history[-2:])
        input_tokens = self._chat_tokenizer(input_text, return_tensors="pt")
        input_tokens.to(self._game_config.device)

        # TODO: can be precomputed
        hallucination_tokens = self._chat_tokenizer(
            # use _chat_tokenizer.s_token
            "<s>" + self._mood_state.hallucinations[self._mood_state.hallucinations_index],
            return_attention_mask=False,
            return_special_tokens_mask=False,
            add_special_tokens=False,
            return_tensors="pt",
        )
        hallucination_tokens.to(self._game_config.device)
        hallucination_input_ids = hallucination_tokens["input_ids"]

        with torch.no_grad():
            outputs = self._chat_model.generate(
                **input_tokens,
                **self._game_config.chat_model_config.generate_kwargs,
                decoder_input_ids=hallucination_input_ids,
                bad_words_ids=self._bad_words_ids,
            )

        response_texts = self._post_process_chat_outputs(
            outputs,
            decoder_input_ids=hallucination_input_ids
        )

        mood_scores = self._get_mood_scores(response_texts)
        sorted_indexes = torch.argsort(-1 * mood_scores) # hack: reverse sort
        chosen_index = sorted_indexes[self._mood_state.mood_priority]
        chosen_response = response_texts[chosen_index]

        return chosen_response
