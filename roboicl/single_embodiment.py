"""Data contract for the 2,700-query, Franka-only main experiment."""

import hashlib
import json
from collections import Counter

PROFILE = "robokata_fi_franka21860_top224_v1"


def audit_plan(plan):
    """Check the training split before constructing caches or batches."""
    if plan.get("data_profile") != PROFILE:
        raise ValueError("Prepare the training cache with prepare.py plan")
    if plan.get("encoding") != "wan_causal_external_rgb_bf16_mu_v1" or plan.get("demo_top_view"):
        raise ValueError("The main recipe requires one external camera for both roles")
    if (plan.get("demo_resize_hw"), plan.get("query_resize_hw")) != ([224, 224], [224, 288]):
        raise ValueError("Expected Demo 224x224 and Query 224x288")
    records = plan["records"]
    pairs = plan["pairs"]
    if len(records) != 3000 or {k: len(v) for k, v in pairs.items()} != {"train": 21860, "validation": 260}:
        raise ValueError("Expected 3000 Franka recordings, 21860 train / 260 validation pairs")
    if plan.get("self_pairs_excluded") != 3000 or plan.get("cross_split_excluded") != 4880:
        raise ValueError("Unexpected original RoboICL pair exclusions")
    if plan.get("split_seed") != 20261105 or len(plan.get("validation_families", [])) != 50:
        raise ValueError("Expected the original seed-20261105 held-out-family split")
    family_counts = Counter(record["family"] for record in records.values())
    if len(family_counts) != 500 or set(family_counts.values()) != {6}:
        raise ValueError("Expected 500 independent families with six task recordings each")
    for episode, record in records.items():
        # Recording IDs identify Franka; explicit metadata must agree when present.
        if not episode.startswith("FR2P_") or any(
            record.get(field, "franka") != "franka" for field in ("kind", "embodiment")
        ):
            raise ValueError(f"Non-Franka recording in main-recipe data: {episode}")
        if "recording_path" in record and not record["recording_path"].lower().endswith(".npz"):
            raise ValueError(f"Expected a Franka NPZ recording: {episode}")
    split_families = {}
    for split, rows in pairs.items():
        seen = set()
        families = set()
        for row in rows:
            demo, query = row["demo"], row["query"]
            if demo == query or (demo, query) in seen:
                raise ValueError("Duplicate or self pair in main-recipe data")
            if records[demo]["family"] == records[query]["family"]:
                raise ValueError("Demo and Query must come from different initial-state families")
            if any(row.get(field, "franka") != "franka" for field in ("demo_embodiment", "query_embodiment")):
                raise ValueError("Cross-embodiment pairs are outside the main recipe")
            seen.add((demo, query))
            for episode in (demo, query):
                record = records[episode]
                if record["task_id"] != row["task_id"]:
                    raise ValueError("Main-recipe pair task mismatch")
                families.add(record["family"])
        split_families[split] = families
    if split_families["train"] & split_families["validation"]:
        raise ValueError("Training/validation families overlap")
    held_out = set(plan["validation_families"])
    if len(held_out) != 50 or not held_out <= set(family_counts):
        raise ValueError("Invalid held-out family list")
    if len(split_families["train"]) != 450 or split_families["train"] & held_out:
        raise ValueError("Expected 450 training families outside the declared holdout")
    # Three of the 50 held-out families have no retained non-self pair in
    # the original plan. Pair coverage need not equal the declared holdout.
    if not split_families["validation"] <= held_out:
        raise ValueError("Validation pairs escape the declared holdout")
    for role in ("demo", "query"):
        if len({pair[role] for pair in pairs["train"]}) != 2700:
            raise ValueError(f"Expected all 2700 training recordings in the {role} role")
    return {
        "schema": "roboicl_franka_only_21860_v1",
        "recordings": len(records),
        "train_pairs": len(pairs["train"]),
        "validation_pairs": len(pairs["validation"]),
        "train_query_recordings": len({p["query"] for p in pairs["train"]}),
        "train_demo_recordings": len({p["demo"] for p in pairs["train"]}),
        "canonical_plan_sha256": hashlib.sha256(
            json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
