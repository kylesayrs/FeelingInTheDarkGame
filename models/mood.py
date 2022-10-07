from typing import List
from pydantic import BaseModel

from config import GameConfig

class MoodState(BaseModel):
    mood_index: int
    mood_priority: int
    hallucinations: List[str]
    hallucinations_index: int

    @classmethod
    def from_game_config(cls, mood_name: str, game_config: GameConfig):
        return cls(
            mood_index=game_config.mood_config[mood_name].mood_index,
            mood_priority=game_config.start_mood_priority,
            hallucinations=game_config.mood_config[mood_name].hallucinations,
            hallucinations_index=game_config.start_hallucinations_index,
        )
