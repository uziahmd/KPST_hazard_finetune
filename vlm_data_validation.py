import json
from pathlib import Path
from collections import Counter, defaultdict

ALLOWED = {
    "hazard_label": {"unsafe_forklift_approach", "no_hazard"},
    "hazard_present": {"yes", "no"},
    "zone_relation": {"outside", "inside", "no_forklift"},
    "object_state": {"stationary", "moving", "no_forklift"},
    "object_direction": {"towards", "away", "none"},
}

REQUIRED_TARGET_KEYS = {
    "hazard_label",
    "hazard_present",
    "zone_relation",
    "object_state",
    "object_direction",
}

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj["_line_num"] = i
            rows.append(obj)
    return rows

def expected_hazard(zone_relation, object_state, object_direction):
    is_hazard = (
        zone_relation == "inside"
        and object_state == "moving"
        and object_direction == "towards"
    )
    if is_hazard:
        return "yes", "unsafe_forklift_approach"
    return "no", "no_hazard"

def validate_target(target):
    violations = []

    missing = REQUIRED_TARGET_KEYS - set(target.keys())
    if missing:
        violations.append(f"missing_keys:{sorted(missing)}")
        return violations

    # Allowed values
    for k, allowed in ALLOWED.items():
        v = target.get(k)
        if v not in allowed:
            violations.append(f"invalid_value:{k}={v!r}")

    zone = target["zone_relation"]
    state = target["object_state"]
    direction = target["object_direction"]
    hazard_present = target["hazard_present"]
    hazard_label = target["hazard_label"]

    # Decision order / consistency rules

    # Rule 1-2: if no forklift is visible
    if zone == "no_forklift" or state == "no_forklift":
        if zone != "no_forklift":
            violations.append("no_forklift_case_zone_must_be_no_forklift")
        if state != "no_forklift":
            violations.append("no_forklift_case_state_must_be_no_forklift")
        if direction != "none":
            violations.append("no_forklift_case_direction_must_be_none")

    # Visible forklift case
    if zone in {"inside", "outside"}:
        if state == "no_forklift":
            violations.append("visible_forklift_case_state_cannot_be_no_forklift")

    if state in {"moving", "stationary"}:
        if zone == "no_forklift":
            violations.append("visible_forklift_case_zone_cannot_be_no_forklift")

    # Motion / direction consistency
    # Stationary forklifts may validly have direction = none.
    if state == "stationary" and direction in {"towards", "away"}:
        violations.append("stationary_forklift_should_not_have_towards_or_away")

    if state == "moving" and direction == "none":
        violations.append("moving_forklift_cannot_have_direction_none")

    # Rule 7-8: hazard must be exactly determined by inside + moving + towards
    exp_present, exp_label = expected_hazard(zone, state, direction)
    if hazard_present != exp_present:
        violations.append(
            f"hazard_present_mismatch:expected_{exp_present}_got_{hazard_present}"
        )
    if hazard_label != exp_label:
        violations.append(
            f"hazard_label_mismatch:expected_{exp_label}_got_{hazard_label}"
        )

    # Positive label must imply the exact positive primitive state
    if hazard_present == "yes":
        if not (zone == "inside" and state == "moving" and direction == "towards"):
            violations.append("positive_label_without_positive_primitives")

    # Negative label must be no_hazard
    if hazard_present == "no" and hazard_label != "no_hazard":
        violations.append("negative_case_must_have_no_hazard_label")

    return violations

def validate_manifest(manifest_path):
    rows = load_jsonl(manifest_path)
    violation_counter = Counter()
    examples = defaultdict(list)

    total = 0
    clean = 0

    for row in rows:
        total += 1
        sample_id = row.get("sample_id", f"line_{row['_line_num']}")
        target = row.get("target", {})
        violations = validate_target(target)

        if not violations:
            clean += 1
        else:
            for v in violations:
                violation_counter[v] += 1
                if len(examples[v]) < 5:
                    examples[v].append({
                        "sample_id": sample_id,
                        "line_num": row["_line_num"],
                        "target": target,
                        "split": row.get("split"),
                        "source_video_id": row.get("source_video_id"),
                    })

    return {
        "manifest_path": str(manifest_path),
        "num_rows": total,
        "num_clean": clean,
        "num_with_violations": total - clean,
        "violation_counts": dict(violation_counter),
        "example_rows": dict(examples),
    }

def main():
    split_report_path = Path("split_report.json")
    report = load_json(split_report_path)

    output_files = report["output_files"]
    manifests = [
        output_files["all_clips_manifest"],
        output_files["train_manifest"],
        output_files["test_manifest"],
    ]

    all_results = []
    for m in manifests:
        manifest_path = Path(m)
        if not manifest_path.exists():
            print(f"[WARN] Missing manifest: {manifest_path}")
            continue
        result = validate_manifest(manifest_path)
        all_results.append(result)

    out_path = Path("label_decision_order_validation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"Saved validation report to: {out_path}\n")

    for result in all_results:
        print("=" * 80)
        print(f"Manifest: {result['manifest_path']}")
        print(f"Rows: {result['num_rows']}")
        print(f"Clean: {result['num_clean']}")
        print(f"Rows with violations: {result['num_with_violations']}")
        if result["violation_counts"]:
            print("Violation counts:")
            for k, v in sorted(result["violation_counts"].items(), key=lambda x: (-x[1], x[0])):
                print(f"  - {k}: {v}")
        else:
            print("No violations found.")

if __name__ == "__main__":
    main()