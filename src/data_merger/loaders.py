def load_ehr(record):
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

    return {
        "instruction": "Will the patient be readmitted? Answer Yes/No/Unknown.",
        "input": ehr_input,
        "output": record.get("diabetes_medication")
    }


def load_research(record):
    return {
        "instruction": "Summarize this medical research paper in simple terms.",
        "input": record["Title"],
        "output": record["Abstract"]
    }


def load_glucose(record):
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

    return {
        "instruction": "Summarize the patient's health metrics for the given day.",
        "input": glucose_input,
        "output": glucose_output
    }


def load_risk(record):
    return {
        "instruction": "Is this patient at risk of cardiovascular disease? Answer 0 = No risk, 1 = At risk.",
        "input": record["input"],
        "output": str(record["label"])
    }
