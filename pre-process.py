import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from pathlib import Path

# ==========================================================
# CONFIGURAÇÕES
# ==========================================================
IMG_DATASET = ["NACT", "ISPY1", "DUKE", "ISPY2" ]


TARGET_SPACING = (1.0, 1.0, 1.0)

# ==========================================================
# FUNÇÃO GENÉRICA DE RESAMPLE
# ==========================================================
def resample_nifti_inplace(nifti_path,target_spacing=(1.0, 1.0, 1.0),interpolation_order=1,output_dtype=np.float32):

    print(f"Processando: {nifti_path}")

    nii = nib.load(str(nifti_path))

    data = nii.get_fdata(dtype=np.float32)

    affine = nii.affine.copy()
    header = nii.header.copy()

    sx, sy, sz = header.get_zooms()[:3]
    tx, ty, tz = target_spacing

    factors = (
        sx / tx,
        sy / ty,
        sz / tz
    )

    print(f"  Spacing original: {(sx, sy, sz)}")
    print(f"  Shape original: {data.shape}")

    data_resampled = zoom(
        data,
        zoom=factors,
        order=interpolation_order
    )

    # Atualiza affine
    affine[0, 0] = np.sign(affine[0, 0]) * tx
    affine[1, 1] = np.sign(affine[1, 1]) * ty
    affine[2, 2] = np.sign(affine[2, 2]) * tz

    new_img = nib.Nifti1Image(
        data_resampled.astype(output_dtype),
        affine,
        header
    )

    new_img.header.set_zooms(target_spacing)

    nib.save(new_img, str(nifti_path))

    print(f"  Shape nova: {data_resampled.shape}")
    print()

# ==========================================================
# IMAGENS
# ==========================================================
def processar_imagens(IMAGES_DIR):

    root = Path(IMAGES_DIR)

    arquivos = sorted(root.rglob("*.nii.gz"))

    print(f"\nImagens encontradas: {len(arquivos)}\n")

    for arquivo in arquivos:

        # ignora fase pré-contraste
        if "_0000.nii.gz" in arquivo.name:
            continue

        try:
            resample_nifti_inplace(
                arquivo,
                target_spacing=TARGET_SPACING,
                interpolation_order=1,
                output_dtype=np.float32
            )

        except Exception as e:
            print(f"Erro em {arquivo}")
            print(e)

# ==========================================================
# MÁSCARAS
# ==========================================================
def processar_mascaras(MASKS_DIR):

    arquivos = sorted(
        Path(MASKS_DIR).glob("*.nii.gz")
    )

    print(f"\nMáscaras encontradas: {len(arquivos)}\n")

    for arquivo in arquivos:

        try:
            resample_nifti_inplace(
                arquivo,
                target_spacing=TARGET_SPACING,
                interpolation_order=0,   # nearest neighbor
                output_dtype=np.uint8
            )

        except Exception as e:
            print(f"Erro em {arquivo}")
            print(e)

# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":

    for dataset in IMG_DATASET:
        print(f"===== RESAMPLE DAS IMAGENS {dataset} =====")
        processar_imagens(f"{dataset}{"/images"}")

        print("===== RESAMPLE DAS MÁSCARAS =====")
        processar_mascaras(f"{dataset}{"/segmentations/expert"}")

    print("Concluído.")