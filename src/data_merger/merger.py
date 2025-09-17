import os, json
from config import INPUT_FOLDER, OUTPUT_JSON, OUTPUT_JSONL
from loaders import load_ehr, load_research, load_glucose, load_risk

def merge_datasets():
    merged = []

    for root, dirs, files in os.walk(INPUT_FOLDER):
        for file in files:
            if not file.endswith(".json"):
                continue

            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                print(f" Skipping invalid JSON file {file_path}: {e}")
                continue

            for record in data:
                if "patient_race" in record:
                    merged.append(load_ehr(record))
                elif "Title" in record and "Abstract" in record:
                    merged.append(load_research(record))
                elif "glucose" in record and "patient" in record:
                    merged.append(load_glucose(record))
                elif "input" in record and "label" in record:
                    merged.append(load_risk(record))
                else:
                    print(f"Unknown schema in file {file}")

    # Save JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    # Save JSONL
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for rec in merged:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Unified dataset saved:")
    print(f" JSON : {OUTPUT_JSON}")
    print(f" JSONL: {OUTPUT_JSONL}")
    print(f"Total records: {len(merged)}")
