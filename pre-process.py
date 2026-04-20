import nibabel as nib
import numpy as np
import cv2
import os
import glob
import SimpleITK as sitk
from skimage.transform import resize


def resample_volume(path_img, path_mask, new_spacing=(1,1,1)):
    # Mantemos o resample físico para garantir que 1px = 1mm em todos os datasets
    img = sitk.ReadImage(path_img)
    mask = sitk.ReadImage(path_mask)

    img = sitk.DICOMOrient(img, "RAS")
    mask = sitk.DICOMOrient(mask, "RAS")

    original_spacing = img.GetSpacing()
    original_size = img.GetSize()

    new_size = [
        int(np.ceil(original_size[i] * (original_spacing[i] / new_spacing[i])))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())

    resampler.SetInterpolator(sitk.sitkLinear)
    img_resampled = resampler.Execute(img)

    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    mask_resampled = resampler.Execute(mask)

    return sitk.GetArrayFromImage(img_resampled), sitk.GetArrayFromImage(mask_resampled)


def process_volume(path_img, path_mask, target_size=(256, 256)):
    # Mantemos o resample 3D para (1,1,1) para garantir padrão físico
    img_vol, mask_vol = resample_volume(path_img, path_mask)
    
    # Z-score normalization (exatamente como no notebook)
    media_vol = np.mean(img_vol)
    desvio_vol = np.std(img_vol)
    img_vol_norm = (img_vol - media_vol) / (desvio_vol + 1e-8)

    processed_slices = []
    depth = img_vol.shape[0] # Volume em (Z, Y, X)
    
    for i in range(depth):
        slice_img = img_vol_norm[i, :, :]
        slice_mask = mask_vol[i, :, :]
        
        # Filtro: processar apenas fatias que contenham a segmentação (tumor)
        if np.sum(slice_mask) > 0:
            
            # REDIMENSIONAMENTO (Substitui Padding e Cropping)
            # preserve_range=True evita que o resize mude a escala do Z-score
            img_final = resize(slice_img, target_size, anti_aliasing=True, preserve_range=True)
            
            # order=0 é CRÍTICO para máscaras: garante que os pixels continuem sendo 0 ou 1
            mask_final = resize(slice_mask, target_size, order=0, anti_aliasing=False, preserve_range=True)

            processed_slices.append((img_final, mask_final))
    
    return processed_slices

# DIRETÓRIOS DO DATASET
DATASET_ROOT = "/home/vitor/Mama-mia"
OUTPUT_DIR = "dados_processados_2d"

os.makedirs(os.path.join(OUTPUT_DIR,"images"),exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR,"masks"),exist_ok=True)

# ENCONTRAR TODAS AS MÁSCARAS
mask_folder = os.path.join(DATASET_ROOT,"segmentations","expert")

mask_files = glob.glob(os.path.join(mask_folder,"*.nii.gz"))

print(f"Encontrados {len(mask_files)} pacientes com segmentação especialista.")

total_fatias_salvas = 0


# LOOP PRINCIPAL DOS PACIENTES
for mask_path in mask_files:
    filename = os.path.basename(mask_path)
    patient_id = filename.replace(".nii.gz","")
    
    # BUSCAR TODAS AS FASES DA MRI (0001, 0002, 0003, 0004...)
    padrao_busca = os.path.join(
        DATASET_ROOT,
        "images",
        patient_id,
        f"{patient_id}_*.nii.gz"
    )
    imagens_paciente = glob.glob(padrao_busca)
    
    for img_path in imagens_paciente:
        nome_arquivo_img = os.path.basename(img_path)
        fase = nome_arquivo_img.replace(".nii.gz","").split("_")[-1]
        
        # Ignorar apenas a fase pre-contrast (0000)
        # Agora ele vai processar 0001, 0002, 0003 e 0004 automaticamente
        if fase != "0001":
            continue
        
        try:
            # Chama a função que criamos no passo anterior com o resize
            slices = process_volume(img_path, mask_path)
            
            for i, (img, mask) in enumerate(slices):
                # Nome do arquivo incluindo a fase para não sobrescrever
                save_name_img = f"{patient_id}_{fase}_slice_{i:03d}.npy"
                
                # A máscara é a mesma para todas as fases, mas salvamos 
                # com o nome da fase para facilitar o carregamento no DataLoader
                save_name_mask = f"{patient_id}_{fase}_slice_{i:03d}_mask.npy"
                
                np.save(os.path.join(OUTPUT_DIR, "images", save_name_img), img)
                np.save(os.path.join(OUTPUT_DIR, "masks", save_name_mask), mask)
                
                total_fatias_salvas += 1
                
            print(f"Processado: {patient_id} Fase: {fase} | Fatias: {len(slices)}")
        
        except Exception as e:
            print(f"[ERRO] Falha ao processar {patient_id} fase {fase}: {e}")


print("\n--- CONCLUÍDO ---")
print(f"Total de fatias prontas para treino: {total_fatias_salvas}")