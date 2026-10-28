"""Check an explicitly configured local tokenizer/template on CPU, without weights."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nice_validation.core import config, write
from nice_validation.data import render, tokenize
from nice_validation.model import load_tokenizer
from nice_validation.protocol import parse_completion, validate_messages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="runs/tokenizer-check.json")
    args = parser.parse_args()
    cfg = config(args.config)
    tokenizer = load_tokenizer(cfg)
    text = "<tool_call>\n<function=bash>\n<parameter=command>\npwd\n</parameter>\n</function>\n</tool_call>"
    parsed = parse_completion(text + "\n" + text, "check", cfg["model"]["thinking"])
    if parsed["parse_error"]:
        raise ValueError(parsed["parse_error"])
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Inspect this development repository."}
            ],
        },
        parsed["message"],
        {"role": "tool", "tool_call_id": "check_0", "content": "FIRST_OBSERVATION"},
        {"role": "tool", "tool_call_id": "check_1", "content": "SECOND_OBSERVATION"},
        {"role": "assistant", "content": "Done."},
    ]
    messages = validate_messages(messages)
    encoded = tokenize(
        tokenizer, messages, cfg["model"]["thinking"], cfg["data"]["max_length"]
    )
    next_prompt = render(tokenizer, messages[:-1], cfg["model"]["thinking"], True)
    write(
        Path(args.output),
        {
            "status": "PASS",
            "weights_loaded": False,
            "weights_downloaded": False,
            "tokenizer_path": cfg["model"]["tokenizer_path"] or cfg["model"]["path"],
            "input_tokens": encoded["input_tokens"],
            "supervised_tokens": encoded["supervised_tokens"],
            "tool_tokens": encoded["tool_tokens"],
            "next_prompt_characters": len(next_prompt),
        },
    )
    print(args.output)


if __name__ == "__main__":
    main()
