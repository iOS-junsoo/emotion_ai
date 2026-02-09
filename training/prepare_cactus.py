from typing import List, Dict, Any
import json
import os


def normalize_cactus(raw_root: str, out_path: str) -> None:
    """
    CACTUS 원본을 읽어 SFT JSONL로 정규화(단순 스텁).
    """
    samples: List[Dict[str, Any]] = [
        {"messages": [{"role": "user", "content": "안녕하세요"}, {"role": "assistant", "content": "안녕하세요. 무엇을 도와드릴까요?"}]}
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Wrote SFT jsonl to {out_path}")


if __name__ == "__main__":
    raw_root = os.environ.get("CACTUS_RAW", "./data/raw/cactus")
    out_path = "./data/processed/sft/train.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    normalize_cactus(raw_root, out_path)


