#!/usr/bin/env python
"""Create or update the Genie space described in genie/space.yml.

    python genie/deploy_genie.py                 # create (or update the space id in genie/.space_id)
    python genie/deploy_genie.py --dry-run       # print the serialized space only
    python genie/deploy_genie.py --ask "What is the median price of entire homes in City A?"

Uses the Databricks CLI (`databricks genie ...`) and your default CLI profile.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
SPACE_ID_FILE = os.path.join(HERE, ".space_id")


def _id(*parts: str) -> str:
    """Stable 32-hex ids so re-deploys update the same instructions/examples."""
    return hashlib.md5("|".join(parts).encode()).hexdigest()


def _lines(text: str) -> list[str]:
    lines = text.strip("\n").split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


def serialized_space(cfg: dict) -> dict:
    space = {
        "version": 2,
        "config": {"sample_questions": [{"id": _id("sample", q), "question": [q]} for q in cfg.get("sample_questions", [])]},
        "data_sources": {"tables": [{"identifier": t} for t in sorted(cfg["tables"])]},
        "instructions": {
            "text_instructions": [{"id": _id("instructions"), "content": _lines(cfg["instructions"])}],
            "example_question_sqls": [
                {"id": _id("example", e["question"]), "question": [e["question"]], "sql": _lines(e["sql"])}
                for e in cfg["example_sqls"]
            ],
            "join_specs": [
                {"id": _id("join", j["left"], j["right"]),
                 "left": {"identifier": j["left"], "alias": j["left"].split(".")[-1]},
                 "right": {"identifier": j["right"], "alias": j["right"].split(".")[-1]},
                 "sql": [j["condition"], f"--rt=FROM_RELATIONSHIP_TYPE_{j['relationship']}--"]}
                for j in cfg.get("joins", [])
            ],
        },
    }
    return space


def cli(*args: str) -> dict:
    out = subprocess.run(["databricks", *args, "-o", "json"], capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"databricks {' '.join(args[:2])} failed: {out.stderr.strip()}")
    return json.loads(out.stdout) if out.stdout.strip() else {}


def warehouse_id(name: str) -> str:
    for w in cli("warehouses", "list"):
        if w["name"] == name:
            return w["id"]
    sys.exit(f"warehouse '{name}' not found")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "space.yml"))
    ap.add_argument("--warehouse", default="Serverless Starter Warehouse")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--ask", help="ask the deployed space a question and print the generated SQL")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    payload = json.dumps(serialized_space(cfg))
    if args.dry_run:
        print(json.dumps(json.loads(payload), indent=2))
        return

    space_id = open(SPACE_ID_FILE).read().strip() if os.path.exists(SPACE_ID_FILE) else ""
    if args.ask:
        if not space_id:
            sys.exit("deploy the space first")
        msg = cli("genie", "start-conversation", space_id, args.ask)
        print(json.dumps(msg, indent=2)[:4000])
        return

    wh = warehouse_id(args.warehouse)
    if space_id:
        cli("genie", "update-space", space_id, "--serialized-space", payload, "--title", cfg["title"],
            "--description", cfg["description"].strip(), "--warehouse-id", wh)
        print("updated Genie space", space_id)
    else:
        res = cli("genie", "create-space", wh, payload, "--title", cfg["title"], "--description", cfg["description"].strip())
        space_id = res["space_id"]
        with open(SPACE_ID_FILE, "w") as f:
            f.write(space_id + "\n")
        print("created Genie space", space_id)


if __name__ == "__main__":
    main()
