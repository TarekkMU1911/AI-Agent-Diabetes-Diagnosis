import os
import json
import uuid
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Load API key
load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("diabetes-diagnosis-db")
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Path to dataset
DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "Datasets", "Input", "UCI_Patients_Preprocessed.json"
)

# Read dataset
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

vectors = []
for i, record in enumerate(data):
    summary = (
        f"Patient record {i}: "
        f"Race {record['patient_race']}, Gender {record['patient_gender']}, "
        f"Age group {record['patient_age_group']}, Hospital stay {record['hospital_stay_days']} days, "
        f"Insurance {record['insurance_type']}, Doctor specialty {record['doctor_specialty']}, "
        f"Lab tests {record['lab_tests_count']}, Procedures {record['procedures_count']}, "
        f"Medications {record['medications_count']}, Outpatient visits {record['past_outpatient_visits']}, "
        f"Emergency visits {record['past_emergency_visits']}, Inpatient visits {record['past_inpatient_visits']}, "
        f"Total diagnoses {record['diagnoses_total']}, Insulin {record['med_insulin']}, "
        f"Medication changed {record['medication_changed']}, "
        f"On diabetes medication {record['diabetes_medication']}, "
        f"Readmission status {record['readmission_status']}."
    )

    vector = model.encode(summary).tolist()
    vector_id = f"uci_record_{i}_{uuid.uuid4().hex[:8]}"

    metadata = {
        "dataset": "UCI",
        "record_type": "patient",
        "race": record["patient_race"],
        "gender": record["patient_gender"],
        "age_group": record["patient_age_group"],
        "doctor_specialty": record["doctor_specialty"],
        "readmission_status": record["readmission_status"]
    }

    vectors.append((vector_id, vector, metadata))

# Upload in batches
batch_size = 100
for i in tqdm(range(0, len(vectors), batch_size)):
    batch = vectors[i:i+batch_size]
    formatted = [{"id": vid, "values": vec, "metadata": meta} for vid, vec, meta in batch]
    index.upsert(vectors=formatted)

print("Upload complete.")
print(index.describe_index_stats())

# Sanity check query
query_text = "Female patient with diabetes readmission risk"
query_vector = model.encode(query_text).tolist()

results = index.query(vector=query_vector, top_k=3, include_metadata=True)

for match in results["matches"]:
    print(f"ID: {match['id']}, Score: {match['score']}")
    print("Metadata:", match['metadata'])
    print()
