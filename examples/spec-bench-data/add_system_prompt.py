#!/usr/bin/env python3
"""
Read question.jsonl and produce question_with_sys.jsonl
where each turn is wrapped with the Llama-3 chat template
(system prompt + user message).
"""

import json, os

SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant. Always answer as helpfully "
    "as possible, while being safe.  Your answers should not include any harmful, "
    "unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure "
    "that your responses are socially unbiased and positive in nature.\n\n"
    "If a question does not make any sense, or is not factually coherent, explain why "
    "instead of answering something not correct. If you don't know the answer to a "
    "question, please don't share false information."
)

def format_llama3(user_msg: str) -> str:
    return (
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_msg}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

def main():
    src = os.path.join(os.path.dirname(__file__), "question.jsonl")
    dst = os.path.join(os.path.dirname(__file__), "question_with_sys.jsonl")

    with open(src, "r", encoding="utf-8") as fin, \
         open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # Apply template to the first turn only (bench uses first turn)
            if "turns" in obj and len(obj["turns"]) > 0:
                obj["turns"][0] = format_llama3(obj["turns"][0])
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Wrote {dst}")

if __name__ == "__main__":
    main()
