import os, json

# Input + Output paths
INPUT_FOLDER = os.path.join(os.path.dirname(__file__), "Input")
OUTPUT_JSON = os.path.join(os.path.dirname(__file__), "diabetes_unified.json")
OUTPUT_JSONL = os.path.join(os.path.dirname(__file__), "diabetes_unified.jsonl")

merged = []

for root, dirs, files in os.walk(INPUT_FOLDER):
    for file in files:
        if file.endswith(".json"):
            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                print(f" Skipping invalid JSON file {file_path}: {e}")
                continue

            for record in data:
                # Dataset 1 → EHR (readmission prediction)
                if "patient_race" in record:
                    ehr_input = (
                        f"Race: {record.get('patient_race')}, "
                        f"Gender: {record.get('patient_gender')}, "
                        f"Age group: {record.get('patient_age_group')}, "
                        f"Hospital stay (days): {record.get('hospital_stay_days')}, "
                        f"Insurance: {record.get('insurance_type')}, "
                        f"Doctor specialty: {record.get('doctor_specialty')}, "
                        f"Lab tests: {record.get('lab_tests_count')}, "
                        f"Procedures: {record.get('procedures_count')}, "
                        f"Medications: {record.get('medications_count')}, "
                        f"readmission status : {record.get('readmission_status')}"
                    )

                    merged.append({
                        "instruction": "Will the patient be readmitted? Answer Yes/No/Unknown.",
                        "input": ehr_input,
                        "output": record.get("diabetes_medication" )
                    })

                # Dataset 2 → Research articles (summarization)
                elif "Title" in record and "Abstract" in record:
                    merged.append({
                        "instruction": "Summarize this medical research paper in simple terms.",
                        "input": record["Title"],
                        "output": record["Abstract"]
                    })

                # Dataset 3 → Patient daily glucose/heart data (descriptive)
                elif "glucose" in record and "patient" in record:
                    glucose_input = (
                        f"Patient: {record['patient']} | Date: {record['date']}\n"
                        f"Glucose (avg/min/max): {record['glucose']['avg']:.1f}/"
                        f"{record['glucose']['min']:.1f}/"
                        f"{record['glucose']['max']:.1f}, "
                        f"Heart rate avg: {record['heart_rate_avg']:.1f}, "
                        f"Steps: {record['steps_total']}, Carbs: {record['carbs_total']}, "
                        f"Insulin: {record['insulin_total']}, "
                        f"Events: Meals={record['events']['meals']}, "
                        f"Activities={record['events']['activity_periods']}, "
                        f"Hypoglycemia={record['events']['hypoglycemia']}, "
                        f"Hyperglycemia={record['events']['hyperglycemia']}"
                    )

                    glucose_output = (
                        f"On {record['date']}, patient {record['patient']} had an average glucose of "
                        f"{record['glucose']['avg']:.1f} (range {record['glucose']['min']:.1f}–{record['glucose']['max']:.1f}), "
                        f"average heart rate {record['heart_rate_avg']:.1f} bpm, "
                        f"{record['steps_total']} steps, {record['carbs_total']} carbs, {record['insulin_total']} insulin units, "
                        f"with {record['events']['meals']} meals, {record['events']['activity_periods']} activity periods, "
                        f"{record['events']['hypoglycemia']} hypoglycemia, and {record['events']['hyperglycemia']} hyperglycemia events."
                    )

                    merged.append({
                        "instruction": "Summarize the patient's health metrics for the given day.",
                        "input": glucose_input,
                        "output": glucose_output
                    })

                # Dataset 4 → Clinical risk classification
                elif "input" in record and "label" in record:
                    merged.append({
                        "instruction": "Is this patient at risk of cardiovascular disease? Answer 0 = No risk, 1 = At risk.",
                        "input": record["input"],
                        "output": str(record["label"])
                    })

                # Unknown schema
                else:
                    print(f"Unknown schema in file {file}")

# Save as JSON (array)
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)

# Save as JSONL (one record per line)
with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
    for rec in merged:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"Unified dataset saved:")
print(f" - JSON : {OUTPUT_JSON}")
print(f" - JSONL: {OUTPUT_JSONL}")
print(f"Total records: {len(merged)}")
