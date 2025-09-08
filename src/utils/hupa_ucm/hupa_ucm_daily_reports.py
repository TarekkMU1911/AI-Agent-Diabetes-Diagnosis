import numpy as np


def summarize_day(group):

    def to_native_type(value):
        if isinstance(value, (np.integer, np.int64, np.int32)):
            return int(value)
        elif isinstance(value, (np.floating, np.float64, np.float32)):
            return float(value)
        elif np.isnan(value):
            return 0.0
        else:
            return value

    return {
        "glucose": {
            "avg": to_native_type(group['glucose'].mean()),
            "min": to_native_type(group['glucose'].min()),
            "max": to_native_type(group['glucose'].max())
        },
        "heart_rate_avg": to_native_type(group['heart_rate'].mean()),
        "steps_total": to_native_type(group['steps'].sum()),
        "carbs_total": to_native_type(group['carb_input'].sum()),
        "insulin_total": to_native_type(group['bolus_volume_delivered'].sum()),
        "events": {
            "meals": to_native_type(group['meal_flag'].sum()),
            "activity_periods": to_native_type(group['activity_flag'].sum()),
            "insulin_periods": to_native_type(group['insulin_flag'].sum()),
            "hypoglycemia": to_native_type((group['glucose'] < 70).sum()),
            "hyperglycemia": to_native_type((group['glucose'] > 180).sum())
        }
    }


def to_text(summary, patient_name, date):
    return (
        f"Patient {patient_name} on {date}: "
        f"average glucose {summary['glucose']['avg']:.1f} mg/dL "
        f"(min {summary['glucose']['min']}, max {summary['glucose']['max']}), "
        f"average heart rate {summary['heart_rate_avg']:.1f} bpm, "
        f"total steps {summary['steps_total']}, "
        f"total carbs {summary['carbs_total']} g, "
        f"total insulin {summary['insulin_total']} units. "
        f"Meals logged: {summary['events']['meals']}, "
        f"physical activity periods: {summary['events']['activity_periods']}, "
        f"insulin doses given: {summary['events']['insulin_periods']}. "
        f"Hypoglycemia events: {summary['events']['hypoglycemia']}, "
        f"Hyperglycemia events: {summary['events']['hyperglycemia']}."
    )


def to_daily_reports(df, patient_name, return_json=False):
    df['date'] = df['time'].dt.date
    texts, jsons = [], []

    for date, group in df.groupby('date'):
        summary = summarize_day(group)
        summary["patient"] = patient_name
        summary["date"] = str(date)

        texts.append(to_text(summary, patient_name, date))
        jsons.append(summary)

    return (texts, jsons) if return_json else texts