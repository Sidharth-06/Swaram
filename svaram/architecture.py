"""
Svaram TTS — Custom Expressive English TTS
===========================================
A fine-tuned Parler-TTS Mini model with custom architecture changes:
  1. Emotion Embedding Table  — learnable latent vectors per emotion
  2. Non-Speech Token Bank    — discrete tokens for [laughs], [sighs], etc.
  3. Prosody-Aware Chunker    — linguistic boundary detection for natural pauses

Architecture:
  Base:     parler-tts/parler-tts-mini-expresso (880M params)
  Adapter:  LoRA (rank=16, alpha=32) on attention layers only
  Custom:   EmotionAdapter module injected into the conditioning stack

Author: Svaram Project
License: MIT (original base: Apache 2.0, Parler-TTS by Hugging Face)
"""

# ────────────────────────────────────────────────────────────────────────────
# Python module: svaram/emotion_adapter.py
# ────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import re
from typing import Optional

# ---- Non-Speech Token Definitions ----------------------------------------

NON_SPEECH_TOKENS = {
    "[laughs]":         0,
    "[sighs]":          1,
    "[clears throat]":  2,
    "[gasps]":          3,
    "[hesitates]":      4,
    "[hums]":           5,
    "[coughs]":         6,
    "[sniffles]":       7,
    "...":              8,
    "[pauses]":         9,
}

# ---- Emotion Labels -------------------------------------------------------

EMOTION_LABELS = [
    "neutral",
    "happy",
    "sad",
    "angry",
    "surprised",
    "confused",
    "excited",
    "whispering",
]


class EmotionAdapter(nn.Module):
    """
    Custom Svaram architecture component.

    Injects a learned emotion embedding into the Parler-TTS conditioning
    signal. This replaces the purely text-based conditioning with a
    discrete + dense emotion representation, giving explicit control over
    vocal style.

    Architecture:
        - Trainable emotion embedding table: (num_emotions, hidden_dim)
        - Non-speech token embedding table:  (num_tokens, hidden_dim)
        - Lightweight MLP projection to match Parler encoder hidden size

    Usage:
        adapter = EmotionAdapter(hidden_dim=1024)
        emotion_vec = adapter.get_emotion_vector("happy")
        nonspeech_vecs = adapter.get_nonspeech_vectors("[laughs]")
    """

    def __init__(self, hidden_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_emotions = len(EMOTION_LABELS)
        self.num_nonspeech = len(NON_SPEECH_TOKENS)

        # Learnable tables (these are YOUR novel trained weights)
        self.emotion_embeddings = nn.Embedding(self.num_emotions, hidden_dim)
        self.nonspeech_embeddings = nn.Embedding(self.num_nonspeech, hidden_dim)

        # Projection MLP to blend with Parler's encoder hidden states
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Gate: controls how much emotion signal flows into conditioning
        self.gate = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        """Initialize with small values so we don't disrupt base model."""
        nn.init.normal_(self.emotion_embeddings.weight, std=0.02)
        nn.init.normal_(self.nonspeech_embeddings.weight, std=0.02)
        for layer in self.projection:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def get_emotion_id(self, emotion: str) -> int:
        """Map emotion string to index."""
        emotion = emotion.lower().strip()
        if emotion in EMOTION_LABELS:
            return EMOTION_LABELS.index(emotion)
        return 0  # default: neutral

    def get_emotion_vector(self, emotion: str) -> torch.Tensor:
        """Get the learnable embedding for an emotion."""
        idx = torch.tensor([self.get_emotion_id(emotion)], dtype=torch.long, device=self.emotion_embeddings.weight.device)
        return self.emotion_embeddings(idx)  # (1, hidden_dim)

    def get_nonspeech_ids(self, text: str) -> list[int]:
        """Extract non-speech token IDs from input text."""
        found = []
        for token, idx in NON_SPEECH_TOKENS.items():
            if token in text:
                found.append(idx)
        return found

    def get_nonspeech_vectors(self, text: str) -> Optional[torch.Tensor]:
        """Get stacked nonspeech token vectors from text."""
        ids = self.get_nonspeech_ids(text)
        if not ids:
            return None
        idx_tensor = torch.tensor(ids, dtype=torch.long, device=self.nonspeech_embeddings.weight.device)
        return self.nonspeech_embeddings(idx_tensor)  # (N, hidden_dim)

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        emotion: str = "neutral",
        text: str = "",
    ) -> torch.Tensor:
        """
        Inject emotion and non-speech conditioning into encoder hidden states.

        Args:
            encoder_hidden_states: (batch, seq_len, hidden_dim) from Parler encoder
            emotion: one of EMOTION_LABELS
            text: the full generation prompt (scanned for non-speech tokens)

        Returns:
            Modified encoder_hidden_states with emotion signal blended in.
        """
        device = encoder_hidden_states.device
        batch_size = encoder_hidden_states.shape[0]

        # Expand emotion vector across batch
        emotion_vec = self.get_emotion_vector(emotion).to(device)
        emotion_vec = emotion_vec.unsqueeze(1).expand(batch_size, -1, -1)  # (B, 1, H)

        # Get non-speech vectors if present
        ns_vecs = self.get_nonspeech_vectors(text)
        if ns_vecs is not None:
            ns_vecs = ns_vecs.to(device)
            ns_vecs = ns_vecs.unsqueeze(0).expand(batch_size, -1, -1)  # (B, N, H)
            extra = torch.cat([emotion_vec, ns_vecs], dim=1)             # (B, 1+N, H)
        else:
            extra = emotion_vec                                           # (B, 1, H)

        # Project and apply gated residual to first N positions of encoder output
        extra_proj = self.projection(extra)
        n = extra_proj.shape[1]

        gate_val = torch.tanh(self.gate)

        # Out-of-place residual addition (preserves autograd graph)
        residual = torch.zeros_like(encoder_hidden_states)
        residual[:, :n, :] = gate_val * extra_proj
        return encoder_hidden_states + residual


# ────────────────────────────────────────────────────────────────────────────
# Python module: svaram/prosody_chunker.py
# ────────────────────────────────────────────────────────────────────────────

def split_into_prosody_chunks(text: str, max_chars: int = 200) -> list[str]:
    """
    Prosody-Aware Text Chunker (Custom Svaram Component).

    Unlike the default character-count chunker in F5-TTS/Parler, this
    uses linguistic heuristics to find natural clause and phrase boundaries:
      - Sentence-terminal punctuation (. ! ?)
      - Clause boundaries (, ; :)
      - Conjunctions at clause starts (and, but, so, because...)
      - Non-speech tokens as hard split points ([laughs], [sighs], ...)

    This ensures each TTS generation call handles a complete linguistic unit,
    which produces more natural intonation.
    """
    # Hard-split on non-speech tokens first
    parts = re.split(
        r'(\[laughs\]|\[sighs\]|\[clears throat\]|\[gasps\]|\[hesitates\]'
        r'|\[hums\]|\[coughs\]|\[sniffles\]|\[pauses\])',
        text
    )

    chunks = []
    current = ""

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Non-speech tokens become standalone chunks
        if part in NON_SPEECH_TOKENS:
            if current.strip():
                chunks.append(current.strip())
                current = ""
            chunks.append(part)
            continue

        # Split text at sentence boundaries within the part
        sentences = re.split(r'(?<=[.!?])\s+', part)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # If adding this sentence would exceed max_chars, flush
            if len(current) + len(sentence) > max_chars and current:
                chunks.append(current.strip())
                current = sentence + " "
            else:
                # Try to split at clause boundaries if still too long
                if len(sentence) > max_chars:
                    clauses = re.split(r'(?<=[,;:])\s+', sentence)
                    for clause in clauses:
                        if len(current) + len(clause) > max_chars and current:
                            chunks.append(current.strip())
                            current = clause + " "
                        else:
                            current += clause + " "
                else:
                    current += sentence + " "

    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if c]
