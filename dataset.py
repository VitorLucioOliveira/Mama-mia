import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import random

class Dataset(Dataset):
    def __init__(self, root_dir, patient_list=None, fase_especifica=None, is_train=False):
        """
        Args:
            root_dir (string): Caminho para a pasta 'dados_processados_2d'
            patient_list (list): Lista de IDs de pacientes permitidos (treino ou teste)
            fase_especifica (string): Filtra uma fase exata (ex: '0001'). None carrega todas.
            is_train (bool): Controla se o Data Augmentation será aplicado nesta instância.
        """
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, "images")
        self.masks_dir = os.path.join(root_dir, "masks")
        self.is_train = is_train  
        
        # Lista apenas imagens (não máscaras) para não duplicar a contagem
        self.files = sorted(glob.glob(os.path.join(self.images_dir, "*.npy")))
        
        if patient_list is not None:
            arquivos_filtrados = []
            for caminho in self.files:
                nome_arquivo = os.path.basename(caminho)
                # Extrai o ID: "DUKE_001_0001_slice_015.npy" -> "DUKE_001"
                partes = nome_arquivo.split('_')
                patient_id = "_".join(partes[:2]) 
                
                if patient_id in patient_list:
                    arquivos_filtrados.append(caminho)
            self.files = arquivos_filtrados

        if fase_especifica is not None:
            self.files = [f for f in self.files if f"_{fase_especifica}_" in f]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        
        img_path = self.files[index]
        filename = os.path.basename(img_path)
        
        # Se a imagem é "DUKE_001_0001_slice_015.npy"
        # A máscara agora se chama "DUKE_001_0001_slice_015_mask.npy"
        mask_filename = filename.replace(".npy", "_mask.npy")
        mask_path = os.path.join(self.masks_dir, mask_filename)
        
        # Carregar do disco
        image = np.load(img_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32)
        
        # Adicionar dimensão de Canal (Channel)
        image = image[None, :, :] 
        mask = mask[None, :, :]
        
        # Converter para Tensor
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)
        
        #  Aleatoriedade isolada (não aumenta dados e Data Augmentation APENAS no Treino)
        if self.is_train:
            # Flip Horizontal (50% de chance)
            if random.random() > 0.5:
                image = torch.flip(image, dims=[2])
                mask = torch.flip(mask, dims=[2])
            
            # Flip Vertical (50% de chance)
            if random.random() > 0.5:
                image = torch.flip(image, dims=[1])
                mask = torch.flip(mask, dims=[1])
            
            # Rotação Aleatória (90, 180 ou 270 graus)
            if random.random() > 0.5:
                k = random.randint(1, 3)
                image = torch.rot90(image, k, dims=[1, 2])
                mask = torch.rot90(mask, k, dims=[1, 2])
        
        return image, mask