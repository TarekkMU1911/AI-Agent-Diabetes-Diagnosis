import os
import json
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Connect to the existing index
index = pc.Index("diabetes-diagnosis-db")

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Base dataset path (relative to project root)
BASE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "Datasets", "HUPA-UCM Diabetes Dataset", "Final", "preprocessed"
)

# Iterate through patients
for patient_folder in os.listdir(BASE_PATH):
    patient_path = os.path.join(BASE_PATH, patient_folder)

    # --- TXT files ---
    text_path = os.path.join(patient_path, "text")
    if os.path.exists(text_path):
        for txt_file in os.listdir(text_path):
            if txt_file.endswith(".txt"):
                file_path = os.path.join(text_path, txt_file)
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()

                vector = model.encode(text).tolist()

                metadata = {
                    "patient": patient_folder,
                    "source_file": txt_file,
                    "type": "text",
                    "dataset": "hupa-ucm"
                }

                vector_id = f"{patient_folder}_{txt_file}"
                index.upsert([(vector_id, vector, metadata)])

    # --- JSON files ---
    json_path = os.path.join(patient_path, "json")
    if os.path.exists(json_path):
        for json_file in os.listdir(json_path):
            if json_file.endswith(".json"):
                file_path = os.path.join(json_path, json_file)
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                for day in data:
                    summary_text = (
                        f"Patient {patient_folder} on {day.get('date','unknown')}: "
                        f"avg glucose {day['glucose']['avg']}, "
                        f"min {day['glucose']['min']}, "
                        f"max {day['glucose']['max']}, "
                        f"heart rate avg {day['heart_rate_avg']}, "
                        f"total steps {day['steps_total']}, "
                        f"total carbs {day['carbs_total']}, "
                        f"total insulin {day['insulin_total']}. "
                        f"Meals: {day['events']['meals']}, "
                        f"activity periods: {day['events']['activity_periods']}, "
                        f"insulin doses: {day['events']['insulin_periods']}, "
                        f"hypoglycemia: {day['events']['hypoglycemia']}, "
                        f"hyperglycemia: {day['events']['hyperglycemia']}."
                    )

                    vector = model.encode(summary_text).tolist()

                    metadata = {
                        "patient": patient_folder,
                        "source_file": json_file,
                        "date": day.get("date", "unknown"),
                        "type": "json",
                        "dataset": "hupa-ucm"
                    }

                    vector_id = f"{patient_folder}_{json_file}_{day.get('date','unknown')}"
                    index.upsert([(vector_id, vector, metadata)])

print(" Finished uploading all patient data to Pinecone.")
