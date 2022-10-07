from typing import List, Dict, Any
from pydantic import BaseModel, Field

import numpy
import torch

#ABC
class MoodConfig(BaseModel):
    pass

class AngryMoodConfig(MoodConfig):
    mood_index: int = 3
    hallucinations: List[str] = [
        ""
    ]
    # TODO: weight_path: str

class ChatModelConfig(BaseModel):
    pretrained_model_name_or_path: str
    generate_kwargs: Dict[str, Any]

class EmotionModelConfig(BaseModel):
    pretrained_model_name_or_path: str

class GameConfig(BaseModel):
    loss_threshold: float = Field(default=numpy.inf)

    start_mood_priority: int = Field(default=10)
    start_hallucinations_index: int = Field(default=0)
    mood_config: Dict[str, MoodConfig] = Field(default={})
    bad_words: List[str] = Field(default=["kill"])

    chat_model_config: ChatModelConfig
    emotion_model_config: EmotionModelConfig

    device: str

    #validation:
    #    assert num_candidate_responses >= max_mood_priority
    #

default_config = GameConfig(
    loss_threshold=5,
    max_mood_priority=5,
    num_candidate_responses=10,
    mood_config={
        "angry": AngryMoodConfig()
    },
    chat_model_config=ChatModelConfig(
        pretrained_model_name_or_path="facebook/blenderbot-400M-distill",
        generate_kwargs={
            "max_length": 30,
            "do_sample": True,
            "top_p": 0.9,
            "temperature": 0.95,
            "num_return_sequences": 5,
        },
    ),
    emotion_model_config=EmotionModelConfig(
        pretrained_model_name_or_path="j-hartmann/emotion-english-distilroberta-base",
    ),
    device=("cuda" if torch.cuda.is_available() else "cpu")
)
