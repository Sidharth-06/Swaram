"""
Svaram TTS — Custom Parler-TTS Architecture
============================================

Author: Svaram Project
License: MIT
"""

from .architecture import (
    EmotionAdapter,
    NON_SPEECH_TOKENS,
    EMOTION_LABELS,
    split_into_prosody_chunks,
)

__all__ = [
    "EmotionAdapter",
    "NON_SPEECH_TOKENS",
    "EMOTION_LABELS",
    "split_into_prosody_chunks",
]
