"""
Data loading utilities.

Loads the instructor-provided Shakespeare dataset.
Each JSON file has structure: {metadata, scenes: [{..., utterances: [...], text, ...}]}

Supports two granularity levels:
- scene: one record per scene (with full scene text, summary, keywords)
- utterance: one record per utterance within each scene
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Literal

from config import PLAY_FILES


Record = Dict[str, Any]


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find dataset file: {path}\n"
            "Place the provided dataset files in data/processed/."
        )
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_scene_records(data: Dict[str, Any]) -> List[Record]:
    """
    Extract scene-level records. Each record contains the full scene text,
    summary, keywords, and metadata.
    """
    records = []
    for scene in data.get("scenes", []):
        records.append({
            "source_id": scene.get("scene_id", ""),
            "play": scene.get("play", ""),
            "act": scene.get("act"),
            "scene": scene.get("scene"),
            "location": scene.get("location", ""),
            "text": scene.get("text", ""),
            "scene_summary": scene.get("scene_summary", ""),
            "keywords": scene.get("keywords", []),
            "num_utterances": len(scene.get("utterances", [])),
        })
    return records


def _extract_utterance_records(data: Dict[str, Any]) -> List[Record]:
    """
    Extract utterance-level records from all scenes.
    """
    records = []
    for scene in data.get("scenes", []):
        scene_meta = {
            "location": scene.get("location", ""),
            "scene_summary": scene.get("scene_summary", ""),
            "keywords": scene.get("keywords", []),
        }
        for utt in scene.get("utterances", []):
            record = {
                "source_id": utt.get("source_id") or utt.get("utterance_id", ""),
                "play": utt.get("play", scene.get("play", "")),
                "act": utt.get("act", scene.get("act")),
                "scene": utt.get("scene", scene.get("scene")),
                "speaker": utt.get("speaker", ""),
                "text": utt.get("text", ""),
            }
            record.update(scene_meta)
            records.append(record)
    return records


def load_play(path: Path, level: Literal["scene", "utterance"] = "scene") -> List[Record]:
    """
    Load one processed Shakespeare JSON file at the specified granularity.
    """
    data = _load_json(path)
    if level == "utterance":
        return _extract_utterance_records(data)
    return _extract_scene_records(data)


def load_all_plays(level: Literal["scene", "utterance"] = "scene") -> List[Record]:
    """
    Load records from all three compulsory plays.
    """
    all_records: List[Record] = []
    for play_key, path in PLAY_FILES.items():
        records = load_play(path, level=level)
        for r in records:
            r.setdefault("play_key", play_key)
        all_records.extend(records)
    return all_records


if __name__ == "__main__":
    for lvl in ("scene", "utterance"):
        records = load_all_plays(level=lvl)
        print(f"[{lvl}] Loaded {len(records)} records.")
        print("First record:")
        print(json.dumps(records[0], indent=2, ensure_ascii=False))
        print()
