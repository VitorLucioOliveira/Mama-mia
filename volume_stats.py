import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path

IMAGES_DIR = "images"

saida = []

for patient_dir in sorted(Path(IMAGES_DIR).iterdir()):

    if not patient_dir.is_dir():
        continue

    for nii_file in patient_dir.glob("*.nii.gz"):

        # ignora pré-contraste
        if "_0000.nii.gz" in nii_file.name:
            continue

        print("Processando:", nii_file)

        img = nib.load(str(nii_file))

        volume = np.asarray(
            img.dataobj,
            dtype=np.float32
        )

        foreground = volume[volume > 0]

        if foreground.size > 0:
            mean = float(foreground.mean())
            std = float(foreground.std())
        else:
            mean = float(volume.mean())
            std = float(volume.std())

        if std < 1e-8:
            std = 1.0

        saida.append({
            "file": str(nii_file),
            "mean": mean,
            "std": std
        })

df = pd.DataFrame(saida)

df.to_csv(
    "volume_stats.csv",
    index=False
)

print(f"\nCSV salvo com {len(df)} volumes.")