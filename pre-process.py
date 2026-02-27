import nibabel as nib
import numpy as np
import cv2
import os
import glob
import SimpleITK as sitk



def resample_volume(path_img, path_mask, new_spacing=(1,1,1)):
    
    # Carregar usando SimpleITK (melhor para manipular spacing físico)
    img = sitk.ReadImage(path_img)
    mask = sitk.ReadImage(path_mask)
    
    # spacing original da imagem (tamanho físico do voxel)
    original_spacing = img.GetSpacing()
    
    # tamanho original do volume
    original_size = img.GetSize()
    
    # calcular novo tamanho baseado no novo spacing
    new_size = [
        int(np.ceil(original_size[i] * (original_spacing[i] / new_spacing[i])))
        for i in range(3)
    ]
    
    # configurar o resampler
    resampler = sitk.ResampleImageFilter()
    
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    
    # imagem usa interpolação linear
    resampler.SetInterpolator(sitk.sitkLinear)
    img_resampled = resampler.Execute(img)
    
    # máscara usa nearest neighbor para não criar labels intermediários
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    mask_resampled = resampler.Execute(mask)
    
    # converter para numpy
    img_np = sitk.GetArrayFromImage(img_resampled)
    mask_np = sitk.GetArrayFromImage(mask_resampled)
    
    # SimpleITK retorna (Z,Y,X)
    # vamos converter para (X,Y,Z) para manter compatível com seu código
    img_np = np.transpose(img_np, (2,1,0))
    mask_np = np.transpose(mask_np, (2,1,0))
    
    return img_np, mask_np



def process_volume(path_img, path_mask, target_size=(256, 256)):
    
    #resample
    img_vol, mask_vol = resample_volume(path_img, path_mask)
    
    #normalização
    media_vol = np.mean(img_vol)
    desvio_vol = np.std(img_vol)
    
    img_vol_norm = (img_vol - media_vol) / (desvio_vol + 1e-8)

    # fatiamento 2D (igual ao seu código)
    processed_slices = []
    depth = img_vol.shape[2]
    
    for i in range(depth):
        
        slice_img = img_vol_norm[:,:,i]
        slice_mask = mask_vol[:,:,i]
        
        
        # manter apenas slices com tumor
        if np.sum(slice_mask) > 0:
            
            # Dados de dimensão 
            target_altura, target_largura = target_size
            img_altura, img_largura = slice_img.shape
            
            # Copia das imagens para resultado
            img_final = slice_img.copy()
            mask_final = slice_mask.copy()
            
            # --- PADDING (Se for menor, preenche com fundo preto) ---
            # Se altura menor adiciona o padding na altura
            if img_altura < target_altura:
                pad_altura =  target_altura - img_altura 
                pad_top = pad_altura // 2
                pad_bottom = pad_altura - pad_top
                
                img_final = np.pad(img_final, ((pad_top, pad_bottom), (0, 0)), mode='constant', constant_values=np.min(img_final))
                mask_final = np.pad(mask_final, ((pad_top, pad_bottom), (0, 0)), mode='constant')
                
            # Se largura menor adiciona o padding na largura
            if img_largura < target_largura:
                pad_largura =  target_largura - img_largura
                pad_rigth = pad_largura // 2
                pad_left = pad_largura - pad_rigth
                
                img_final = np.pad(img_final, ((0, 0),(pad_rigth, pad_left)), mode='constant', constant_values=np.min(img_final))
                mask_final = np.pad(mask_final, ((0, 0),(pad_rigth, pad_left)), mode='constant')
                
            # ATUALIZANDO para nao tirar a proporção 
            img_altura, img_largura = img_final.shape
                         
            # --- CROPPING (Se for maior, corta o centro) --------
            # Descobre as coordenadas Y e X de todos os pixels do tumor
            y_tumor, x_tumor = np.where(mask_final > 0)
            
            # Acha o "centro de gravidade" do tumor
            centro_tumor_y = int(np.mean(y_tumor))
            centro_tumor_x = int(np.mean(x_tumor))

            # Se altura maior, calcula o corte para deixar o tumor no meio (sem sair da borda da imagem)
            if img_altura > target_altura:
                corte_y = max(0, min(centro_tumor_y - (target_altura // 2), img_altura - target_altura))
                img_final = img_final[corte_y : corte_y + target_altura, :]
                mask_final = mask_final[corte_y : corte_y + target_altura, :]
                
            # Se largura maior, calcula o corte para deixar o tumor no meio (sem sair da borda)
            if img_largura > target_largura:
                corte_x = max(0, min(centro_tumor_x - (target_largura // 2), img_largura - target_largura))
                img_final = img_final[ : , corte_x : corte_x + target_largura ]
                mask_final = mask_final[ : , corte_x : corte_x + target_largura ]
            processed_slices.append((img_final, mask_final))
    
    
    return processed_slices

DATASET_ROOT = "/home/vitor/Mama-mia" 
OUTPUT_DIR = "dados_processados_2d"

os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "masks"), exist_ok=True)

mask_folder = os.path.join(DATASET_ROOT, "segmentations", "expert")
mask_files = glob.glob(os.path.join(mask_folder, "DUKE_*.nii.gz"))

print(f"Encontrados {len(mask_files)} pacientes com segmentação especialista.")


total_fatias_salvas = 0

for mask_path in mask_files:
    
    filename = os.path.basename(mask_path)
    patient_id = filename.replace(".nii.gz", "")
    
    # 1. Rastreador Dinâmico: Busca qualquer arquivo que siga o padrão do paciente
    padrao_busca = os.path.join(DATASET_ROOT, "images", patient_id, f"{patient_id}_*.nii.gz")
    imagens_paciente = glob.glob(padrao_busca)
    
    # Prevenção: Se a pasta estiver vazia, avisa e pula para o próximo paciente
    if len(imagens_paciente) == 0:
        print(f"[AVISO] Nenhuma imagem encontrada para {patient_id}. Pulando.")
        continue
    
    # 2. Itera diretamente sobre os caminhos reais encontrados no HD
    for img_path in imagens_paciente: 
        
        # 3. Descobrindo a Fase: Extrai o nome do arquivo e pega a última parte após o "_"
        # Ex: "DUKE_001_0002.nii.gz" -> "DUKE_001_0002" -> ["DUKE", "001", "0002"] -> "0002"
        nome_arquivo_img = os.path.basename(img_path)
        fase = nome_arquivo_img.replace(".nii.gz", "").split("_")[-1]
        
        if fase == "0000":
            continue  # Ignora esta iteração e pula para a próxima imagem
        
        try:
            
            slices = process_volume(img_path, mask_path)
            
            for i, (img, mask) in enumerate(slices):
                
                save_name_img = f"{patient_id}_{fase}_slice_{i:03d}.npy"
                save_name_mask = f"{patient_id}_slice_{i:03d}.npy"
                
                np.save(os.path.join(OUTPUT_DIR, "images", save_name_img), img)
                
                if not os.path.exists(os.path.join(OUTPUT_DIR, "masks", save_name_mask)):
                    np.save(os.path.join(OUTPUT_DIR, "masks", save_name_mask), mask)
                
                total_fatias_salvas += 1
                
                
            print(f"Processado: {patient_id}_{fase} | Fatias salvas: {len(slices)}")
        
        except Exception as e:
            print(f"[ERRO] Falha ao processar {patient_id}_{fase} : {e}")
    

print(f"\n--- CONCLUÍDO ---")
print(f"Total de fatias prontas para treino: {total_fatias_salvas}")