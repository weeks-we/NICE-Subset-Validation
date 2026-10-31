"""Inspect a bounded number of real records before selecting an action protocol."""

import argparse
import itertools
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--rows", type=int, default=3)
    args = parser.parse_args()
    with open(args.source, encoding="utf-8") as handle:
        for line in itertools.islice(handle, args.rows):
            row = json.loads(line)
            messages = row["messages"]
            messages = json.loads(messages) if isinstance(messages, str) else messages
            calls = [
                call["function"]
                for msg in messages
                for call in msg.get("tool_calls") or []
            ]
            print(
                json.dumps(
                    {
                        "instance_id": row["instance_id"],
                        "traj_id": row.get("traj_id"),
                        "columns": sorted(row),
                        "roles": [msg["role"] for msg in messages],
                        "message_fields": sorted(
                            {key for msg in messages for key in msg}
                        ),
                        "tools": [
                            {"name": call["name"], "arguments": call["arguments"]}
                            for call in calls
                        ],
                        "resolved": row.get("resolved"),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
