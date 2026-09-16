import pandas as pd

from src.constants import BINARY_TARGET_MAP, DATA_DIR, MULTI_TARGET_MAP
from src.utils import get_project_relative_path

RAW_DATA_DIR = DATA_DIR / "raw" / "chest_xray"

records = []
for img_path in RAW_DATA_DIR.rglob("*.jpeg"):
    rel_path = get_project_relative_path(img_path)
    filename = img_path.name
    patient_id = filename.split("-")[1]
    if "NORMAL" in str(img_path):
        c = "NORMAL"
        subc = "NORMAL"
    else:
        c = "PNEUMONIA"
        subc = "BACTERIA" if "bacteria" in filename.lower() else "VIRUS"

    records.append(
        {
            "filepath": rel_path,
            "filename": filename,
            "patient_id": patient_id,
            "class": c,
            "subclass": subc,
        }
    )

df = pd.DataFrame(records)
df["label"] = df["class"].map(BINARY_TARGET_MAP)
df["sublabel"] = df["subclass"].map(MULTI_TARGET_MAP)

output_dir = DATA_DIR / "metadata"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "catalog.csv"

df.to_csv(output_path, index=False)
print(
    f"Catalog saved to {output_path}:\n{len(df)} images across {df['patient_id'].nunique()} unique patients."
)
