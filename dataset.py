import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import random

class DukeDataset(Dataset):
    def __init__(self, root_dir, patient_list=None):
        """
        Args:
            root_dir (string): Caminho para a pasta 'dados_processados_2d'
        """
     
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, "images")
        self.masks_dir = os.path.join(root_dir, "masks")
        
        # 1. Listar os arquivos .npy da pasta images
        # O glob pega todos os caminhos. O sorted para garantir ordem.
        self.files = sorted(glob.glob(os.path.join(self.images_dir, "*.npy")))
        
        # 2. Verificar pacientes
        if patient_list is not None:
            arquivos_filtrados= []
            
            # percorre todos os arquivos
            for caminho in self.files:
                
                nome_arquivo = os.path.basename(caminho)
                patient_id = nome_arquivo[:8]
                
                # Se o paciente estiver na lista desejada (treino/test) insere na nova lista
                if patient_id in patient_list:
                    arquivos_filtrados.append(caminho)
                    
            self.files = arquivos_filtrados
            
        # 3. Verificação de segurança
        if len(self.files) == 0:
            print(f"ERRO: Nenhuma imagem encontrada em {self.images_dir}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        # Achar o arquivo com o index pedido
        img_path = self.files[index]
        
        # Descobrir qual é a máscara correspondente
        filename = os.path.basename(img_path)
        mask_filename = filename[:8] + filename[13:] # Pula a fase da imagem do paciente (DUKE_001|_0002|_slice_015.npy)
        mask_path = os.path.join(self.masks_dir, mask_filename)
        
        # Carregar do disco (Numpy)
        # .astype(np.float32) é vital: Redes neurais usam float32. 
        image = np.load(img_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32)
        
        # Adicionar dimensão de Canal (Channel)
        # Imagem atual é (256, 256) -> Altura x Largura
        # O PyTorch exige (1, 256, 256) -> Canais x Altura x Largura
        image = image[None, :, :] 
        mask = mask[None, :, :]
        
        # Converter para Tensor
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)
        
        # Random flipping
        if random.random() < 0.5:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[2])
        
        return image, mask
    pass

