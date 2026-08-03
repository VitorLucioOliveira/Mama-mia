"""MAMA-MIA federado: modelo, loss, métricas e funções de treino/avaliação.

Tudo portado fielmente do treinamento centralizado (train.py):
  - LightweightUNet (DepthwiseSeparableConv + InstanceNorm), exposta como `Net`
  - DiceBCELoss (metade Dice, metade BCE)
  - Dice 2D por batch (monitoramento do treino)
  - Dice volumétrico 3D (avaliação, remontando o volume por paciente)
"""

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset import MamaMia3DOnTheFlyDataset



def set_seed(seed: int = 42):
    """Fixa todas as fontes de aleatoriedade para reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
# ============================================================
# Arquitetura: Lightweight U-Net com InstanceNorm (Edge/ARM)
# ============================================================
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1,
            groups=in_channels, bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        return self.relu(x)


class LightweightUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[16, 32, 64, 128]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(
                nn.Sequential(
                    DepthwiseSeparableConv(in_channels, feature),
                    DepthwiseSeparableConv(feature, feature),
                )
            )
            in_channels = feature

        self.bottleneck = nn.Sequential(
            DepthwiseSeparableConv(features[-1], features[-1] * 2),
            DepthwiseSeparableConv(features[-1] * 2, features[-1] * 2),
        )

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2, bias=False)
            )
            self.ups.append(
                nn.Sequential(
                    DepthwiseSeparableConv(feature * 2, feature),
                    DepthwiseSeparableConv(feature, feature),
                )
            )

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)


# O Flower (client_app / server_app) importa a rede pelo nome `Net`.
Net = LightweightUNet


# ============================================================
# Loss e métricas
# ============================================================
class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice_loss = 1 - (2.0 * intersection + self.smooth) / (union + self.smooth)
        return (self.bce_weight * bce_loss) + (self.dice_weight * dice_loss.mean())


def calculate_real_dice(logits, targets, smooth=1.0):
    """Dice 2D por batch, apenas para monitorar o treino."""
    preds = (logits > 0.0).float()
    intersection = (preds * targets).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean().item()


def calculate_volume_dice(pred_volume, gt_volume, smooth=1.0):
    """Dice volumétrico 3D: recebe o volume inteiro de um paciente."""
    pred_volume = pred_volume.astype(np.float32)
    gt_volume = gt_volume.astype(np.float32)
    intersection = (pred_volume * gt_volume).sum()
    dice = (2.0 * intersection + smooth) / (
        pred_volume.sum() + gt_volume.sum() + smooth
    )
    return float(dice)


# ============================================================
# Leitura dos caminhos e carregamento dos dados locais do cliente
# ============================================================
def obter_caminhos_por_lista(lista_pacientes, pasta_imagens, pasta_mascaras, train):
    """Encontra imagens e máscaras de cada paciente (Pathlib, de forma segura)."""
    img_paths = []
    mask_paths = []
    dir_img = Path(pasta_imagens)
    dir_mask = Path(pasta_mascaras)

    for patient_id in lista_pacientes:
        patient_id = str(patient_id).strip()
        pasta_paciente = dir_img / patient_id

        # if not pasta_paciente.is_dir():
        #     print(f"Aviso: Pasta de imagem não encontrada -> {pasta_paciente}")
        #     continue

        mask_file = dir_mask / f"{patient_id}.nii"
        # if not mask_file.exists():
        #     print(f"Aviso: Máscara não encontrada para -> {patient_id}")
        #     continue

        # if not train:
        #     fases = [f for f in pasta_paciente.glob("*.nii") if "0001" in f.name]
        # else:
        #     fases = [f for f in pasta_paciente.glob("*.nii") if "0000" not in f.name]
        
        fases = [f for f in pasta_paciente.glob("*.nii") if "0001" in f.name]

        # if not fases:
        #     print(f"Aviso: A pasta {patient_id} está vazia (sem .nii)")
        #     continue

        for img_file in fases:
            img_paths.append(str(img_file))
            mask_paths.append(str(mask_file))

    return img_paths, mask_paths


def load_data_train(data_dir: str, batch_size: int):
    """Carrega os dados locais deste cliente e devolve (trainloader, valloader).

    Cada cliente faz o seu próprio split 80/20 local sobre os pacientes da sua
    coluna `train_split`. O valloader usa shuffle=False de propósito, para que as
    fatias cheguem agrupadas por volume e o Dice 3D possa ser remontado.
    """
    pasta_imagens = os.path.join(data_dir, "images")
    pasta_mascaras = os.path.join(data_dir, "segmentations", "expert")
    arquivo_csv = os.path.join(data_dir, "train_test_splits.csv")
    arquivo_csv_stats = os.path.join(data_dir, "volume_stats.csv")

    df_splits = pd.read_csv(arquivo_csv)
    pool_treino = df_splits["train_split"].dropna().astype(str).str.strip().tolist()

    pacientes_train, _ = train_test_split(
        pool_treino, test_size=0.2, random_state=42, shuffle=True
    )

    img_train, mask_train = obter_caminhos_por_lista(
        pacientes_train, pasta_imagens, pasta_mascaras, train=True
    )

    train_dataset = MamaMia3DOnTheFlyDataset(
        img_train, mask_train, stats_csv=arquivo_csv_stats
    )


    trainloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False, persistent_workers=False,
    )


    return trainloader


def load_data_val(data_dir: str, batch_size: int):
    """Carrega os dados locais deste cliente e devolve (trainloader, valloader).

    Cada cliente faz o seu próprio split 80/20 local sobre os pacientes da sua
    coluna `train_split`. O valloader usa shuffle=False de propósito, para que as
    fatias cheguem agrupadas por volume e o Dice 3D possa ser remontado.
    """
    pasta_imagens = os.path.join(data_dir, "images")
    pasta_mascaras = os.path.join(data_dir, "segmentations", "expert")
    arquivo_csv = os.path.join(data_dir, "train_test_splits.csv")
    arquivo_csv_stats = os.path.join(data_dir, "volume_stats.csv")

    df_splits = pd.read_csv(arquivo_csv)
    pool_treino = df_splits["train_split"].dropna().astype(str).str.strip().tolist()

    _, pacientes_val = train_test_split(
        pool_treino, test_size=0.2, random_state=42, shuffle=True
    )

  
    img_val, mask_val = obter_caminhos_por_lista(
        pacientes_val, pasta_imagens, pasta_mascaras, train=True
    )

   
    val_dataset = MamaMia3DOnTheFlyDataset(
        img_val, mask_val, stats_csv=arquivo_csv_stats
    )

    valloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False, persistent_workers=False,
    )

    return valloader



# ============================================================
# Treino e avaliação locais
# ============================================================
def train(net, trainloader, epochs, lr, device, accumulation_steps=4):
    """Treina o modelo localmente por `epochs` épocas (com acúmulo de gradiente)."""
    net.to(device)
    criterion = DiceBCELoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()

    running_loss = 0.0
    n_batches = 0

    for _ in range(epochs):
        optimizer.zero_grad()
        for i, (images, masks, _, _) in enumerate(trainloader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = net(images)
            loss = criterion(outputs, masks)
            loss = loss / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(trainloader):
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item() * accumulation_steps
            n_batches += 1

    return running_loss / max(n_batches, 1)


def test(net, valloader, device):
    """Avalia o modelo com Dice VOLUMÉTRICO 3D (streaming por paciente).

    As fatias chegam agrupadas por volume (valloader tem shuffle=False). Ao trocar
    de volume, empilha as fatias acumuladas, calcula o Dice 3D e limpa a RAM.
    """
    net.to(device)
    criterion = DiceBCELoss().to(device)
    net.eval()

    val_loss = 0.0
    volume_dices = []

    current_vol_id = None
    current_pred_slices = {}
    current_gt_slices = {}

    with torch.no_grad():
        for images, masks, vol_idxs, z_idxs in valloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = net(images)
            val_loss += criterion(outputs, masks).item()

            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
            masks_np = masks.cpu().numpy()
            vol_idxs_np = vol_idxs.numpy()
            z_idxs_np = z_idxs.numpy()

            for b in range(len(vol_idxs_np)):
                vol_id = int(vol_idxs_np[b])
                z_id = int(z_idxs_np[b])

                if current_vol_id is None:
                    current_vol_id = vol_id

                # Trocou de paciente? Fecha o volume anterior e calcula o Dice 3D
                if vol_id != current_vol_id:
                    ordered = sorted(current_pred_slices.keys())
                    pred_volume = np.stack([current_pred_slices[z] for z in ordered])
                    gt_volume = np.stack([current_gt_slices[z] for z in ordered])
                    volume_dices.append(calculate_volume_dice(pred_volume, gt_volume))

                    current_pred_slices.clear()
                    current_gt_slices.clear()
                    current_vol_id = vol_id

                current_pred_slices[z_id] = preds[b, 0]
                current_gt_slices[z_id] = masks_np[b, 0]

    # Fecha o último paciente que ficou no buffer após o loop
    if current_pred_slices:
        ordered = sorted(current_pred_slices.keys())
        pred_volume = np.stack([current_pred_slices[z] for z in ordered])
        gt_volume = np.stack([current_gt_slices[z] for z in ordered])
        volume_dices.append(calculate_volume_dice(pred_volume, gt_volume))

    avg_loss = val_loss / len(valloader)
    avg_dice = float(np.mean(volume_dices)) if volume_dices else 0.0
    return avg_loss, avg_dice