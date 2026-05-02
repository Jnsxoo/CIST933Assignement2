"""
Configuration for the Assignment 2 starter code.

Students should adjust these values to match their own implementation.
"""

import os
import sys

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["USE_TF"] = "0"

# Block broken h5py / TensorFlow from being imported by transformers
import types
_fake = types.ModuleType("h5py")
_fake.__version__ = "0.0.0"
sys.modules["h5py"] = _fake

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
PROMPT_DIR = PROJECT_ROOT / "prompts"
RESULTS_DIR = PROJECT_ROOT / "results"

PLAY_FILES = {
    "hamlet": DATA_DIR / "hamlet.json",
    "macbeth": DATA_DIR / "macbeth.json",
    "romeo_and_juliet": DATA_DIR / "romeo_and_juliet.json",
}

DEFAULT_TOP_K = 3

# Suggested lightweight embedding model.
# Students may change this and justify the choice in the report.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Generation model — a small, locally-runnable language model.
# TinyLlama-1.1B-Chat is chosen for its balance of quality and resource efficiency.
GENERATION_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
GENERATION_MAX_NEW_TOKENS = 512
GENERATION_TEMPERATURE = 0.7

# Data loading granularity: "scene" or "utterance"
CHUNK_LEVEL = "scene"
