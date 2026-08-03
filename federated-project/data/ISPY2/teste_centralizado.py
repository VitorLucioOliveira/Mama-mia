import os
import gc
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

# Importamos a função calculate_volume_dice do train.py
from task import (
    Net,
    calculate_volume_dice,
    obter_caminhos_por_lista,
)
from dataset import MamaMia3DOnTheFlyDataset
# ============================================================
# 1. CONFIGURAÇÕES GERAIS
# ============================================================

torch.backends.cudnn.benchmark = False
os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'

# ------------------------------------------------------------------
# MODO DE SELEÇÃO DAS FATIAS
#
# "selecionadas" -> mantém o filtro do Dataset (fatias com lesão + vizinhas
#                   imediatas + uma amostra das fatias vazias, na proporção 1:2).
#                   É o MESMO critério usado no treino e na validação.
#
# "completo"     -> avalia TODAS as fatias do volume anatômico, sem filtro.
#                   Mais rigoroso e mais próximo do uso clínico real: o modelo
#                   enfrenta todas as regiões onde pode gerar falso positivo.
#                   Os valores tendem a ser MENORES que no modo "selecionadas".
#
# Importante: para comparar centralizado x federado, use o MESMO modo nos dois.
# ------------------------------------------------------------------


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# ============================================================
# 2. MÉTRICAS (Volume 3D)
# ============================================================

# O Volume Dice já é importado do train.py, 
# mas precisamos de uma versão Volumétrica para o IoU:
def calculate_volume_iou(pred_volume, gt_volume, smooth=1.0):
    pred_volume = pred_volume.astype(np.float32)
    gt_volume = gt_volume.astype(np.float32)

    intersection = (pred_volume * gt_volume).sum()
    union = pred_volume.sum() + gt_volume.sum() - intersection

    iou = (intersection + smooth) / (union + smooth)
    return float(iou)


def calculate_volume_metrics(pred_volume, gt_volume, smooth=1e-6):
    """
    Calcula Dice, IoU, Precision e Recall para um volume 3D.
    """

    pred = pred_volume.astype(bool)
    gt = gt_volume.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()

    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)

    iou = (tp + smooth) / (tp + fp + fn + smooth)

    precision = (tp + smooth) / (tp + fp + smooth)

    recall = (tp + smooth) / (tp + fn + smooth)

    return float(dice), float(iou), float(precision), float(recall)
# ============================================================
# 3. TESTE
# ============================================================

def test_model(modo_fatias):
    MODO_FATIAS = modo_fatias   
    set_seed(42)
    device = torch.device("cpu")

    # ========================================================
    # CONFIGS
    # ========================================================
    pasta_imagens = "images"
    pasta_mascaras = "segmentations/expert"
    arquivo_csv = "train_test_splits.csv"
    arquivo_csv_stats = "volume_stats.csv"  # <-- Adicionado para match com o train.py
    modelo_path = "best_lightweight_unet_arm_ispy2.pth"
    batch_size = 2

    # ========================================================
    # LEITURA DO CSV
    # ========================================================
    print(f"Lendo {arquivo_csv}...")
    df_splits = pd.read_csv(arquivo_csv)

    pacientes_test = (
        df_splits['test_split']
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    print(f"\nPacientes de teste: {len(pacientes_test)}")

    # ========================================================
    # CAMINHOS
    # ========================================================
    img_test, mask_test = obter_caminhos_por_lista(
        pacientes_test,
        pasta_imagens,
        pasta_mascaras,
        train=False
    )

    print(f"Volumes de teste: {len(img_test)}")

    # ========================================================
    # DATASET / DATALOADER
    # ========================================================
    test_dataset = MamaMia3DOnTheFlyDataset(
        img_test,
        mask_test,
        stats_csv=arquivo_csv_stats  
    )

    # ------------------------------------------------------------------
    # MODO "COMPLETO": reconstrói o mapa de fatias para incluir TODAS as
    # fatias de cada volume (z = 0 .. Z-1), ignorando o filtro do __init__.
    # `proxies[v][0]` é o proxy NIfTI da imagem do volume v; .shape[2] é o
    # número de fatias no eixo Z.
    # ------------------------------------------------------------------
    if MODO_FATIAS == "completo":
        test_dataset.slice_map = [
            (v, z)
            for v in range(len(test_dataset.proxies))
            for z in range(test_dataset.proxies[v][0].shape[2])
        ]

    print(f"Fatias de teste ({MODO_FATIAS}): {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # ========================================================
    # MODELO
    # ========================================================
    model = Net(in_channels=1, out_channels=1)

    model.load_state_dict(
        torch.load(
            modelo_path,
            map_location=device,
            weights_only=True
        )
    )

    # Convertendo para channels_last igual ao train.py para CPU performance
    model = model.to(memory_format=torch.channels_last).to(device)
    model.eval()

    print(f"\nModelo carregado: {modelo_path}")

    # ========================================================
    # LÓGICA DE TESTE COM STREAMING VOLUMÉTRICO
    # ========================================================
    volume_dices = []
    volume_ious = []
    volume_precisions = []
    volume_recalls = []
    
    print("\nIniciando teste...\n")

    with torch.no_grad():
        current_vol_id = None
        current_pred_slices = {}
        current_gt_slices = {}

        # O Dataloader agora retorna 4 variáveis
        for images, masks, vol_idxs, z_idxs in tqdm(test_loader, desc="Testando Volumes"):

            # Formatando a imagem para inference em CPU ARM otimizado
            images = images.to(device, memory_format=torch.channels_last)
            masks = masks.to(device)

            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
            masks_np = masks.cpu().numpy()
            
            vol_idxs_np = vol_idxs.numpy()
            z_idxs_np = z_idxs.numpy()

            for b in range(len(vol_idxs_np)):
                vol_id = int(vol_idxs_np[b])
                z_id = int(z_idxs_np[b])

                # 1. Inicializa o rastreador
                if current_vol_id is None:
                    current_vol_id = vol_id

                # 2. GATILHO: Quando muda o volume, calcula as métricas do anterior
                if vol_id != current_vol_id:
                    ordered_slices = sorted(current_pred_slices.keys())
                    pred_volume = np.stack([current_pred_slices[z] for z in ordered_slices])
                    gt_volume = np.stack([current_gt_slices[z] for z in ordered_slices])
                    
                    dice, iou, precision, recall = calculate_volume_metrics(
                        pred_volume,
                        gt_volume
                    )
                    
                    volume_dices.append(dice)
                    volume_ious.append(iou)
                    volume_precisions.append(precision)
                    volume_recalls.append(recall)
                    
                    current_pred_slices.clear()
                    current_gt_slices.clear()
                    current_vol_id = vol_id

                # 3. Empilha a fatia atual na memória
                current_pred_slices[z_id] = preds[b, 0]
                current_gt_slices[z_id] = masks_np[b, 0]

            del images, masks, outputs
            gc.collect()

        # GATILHO FINAL: Processa o último volume preso no buffer ao final do loop
        if current_pred_slices:
            ordered_slices = sorted(current_pred_slices.keys())
            pred_volume = np.stack([current_pred_slices[z] for z in ordered_slices])
            gt_volume = np.stack([current_gt_slices[z] for z in ordered_slices])
            
            dice, iou, precision, recall = calculate_volume_metrics(
                pred_volume,
                gt_volume
            )

            volume_dices.append(dice)
            volume_ious.append(iou)
            volume_precisions.append(precision)
            volume_recalls.append(recall)

    # ========================================================
    # MÉDIAS E DESVIOS PADRÃO
    # ========================================================
    media_test_dice = np.mean(volume_dices)
    std_test_dice   = np.std(volume_dices)
    media_test_iou  = np.mean(volume_ious)
    std_test_iou    = np.std(volume_ious)
    media_precision = np.mean(volume_precisions)
    std_precision = np.std(volume_precisions)

    media_recall = np.mean(volume_recalls)
    std_recall = np.std(volume_recalls)

    # ========================================================
    # RESULTADOS
    # ========================================================
    print("\n================ RESULTADOS =================")
    print(f"Modo de fatias   : {MODO_FATIAS}")
    print(f"Volumes Testados : {len(volume_dices)}")
    print(f"Test 3D Dice     : {media_test_dice:.4f} +/- {std_test_dice:.4f}")
    print(f"Test 3D IoU      : {media_test_iou:.4f} +/- {std_test_iou:.4f}")
    print(f"Test Precision  : {media_precision:.4f} +/- {std_precision:.4f}")
    print(f"Test Recall     : {media_recall:.4f} +/- {std_recall:.4f}")
    print("=============================================\n")

    # ========================================================
    # SALVAR CSV
    # ========================================================


    # Um Dice/IoU por paciente -- necessario para testes estatisticos
    # (ex.: comparar centralizado x federado com Wilcoxon pareado).
    pd.DataFrame({
        "Modo_Fatias": MODO_FATIAS,
        "dice": volume_dices,
        "iou": volume_ious,
        "precision": volume_precisions,
        "recall": volume_recalls
    }).to_csv(f"test_results_por_volume_{modo_fatias}.csv", index=False)

    print("Resultados salvos em: test_results.csv")
    print("Dice/IoU por volume : test_results_por_volume.csv")
    
    return {
    "Modo_Fatias": MODO_FATIAS,
    "Dice": media_test_dice,
    "Dice_Std": std_test_dice,
    "IoU": media_test_iou,
    "IoU_Std": std_test_iou,
    "Precision": media_precision,
    "Precision_Std": std_precision,
    "Recall": media_recall,
    "Recall_Std": std_recall,
    "Volumes": len(volume_dices)
}
    

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    resultados = []
    resultados.append(test_model("selecionadas"))
    resultados.append(test_model("completo"))

    pd.DataFrame(resultados).to_csv("test_results.csv", index=False)
