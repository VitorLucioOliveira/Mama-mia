"""Avalia um modelo no conjunto de TESTE de cada dataset, separadamente.

Roda no host (sem Docker, sem SuperLink) — é só inferência sobre o modelo salvo.
Para cada cliente (DUKE, ISPY1, ISPY2, NACT), usa a coluna `test_split` do CSV local
(nunca vista no treino) e calcula Dice e IoU VOLUMÉTRICOS 3D por paciente.

Reporta, por dataset: média ± desvio padrão entre os volumes.
"""

import gc
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Importa do MESMO pacote que os clientes federados usaram (já corrigido para .nii)
from task import (
    Net,
    obter_caminhos_por_lista,
)
from dataset import MamaMia3DOnTheFlyDataset

# ============================================================
# 1. CONFIGURAÇÕES
# ============================================================
BASE_DIR = "data"                      # pasta que contém DUKE/, ISPY1/, ISPY2/, NACT/
CLIENTES = ["DUKE", "ISPY1", "ISPY2", "NACT"]
MODELO_PATH = ["final_model_mamamia_uni.pt"]         # modelo global do federado
                                       # (troque pelo .pth do centralizado para comparar)
BATCH_SIZE = 2
SAIDA_CSV = "test_results.csv"             # resumo por dataset
SAIDA_CSV_VOLUMES = "test_results_por_volume.csv"  # um Dice/IoU por paciente

# "selecionadas" = mesmo critério de fatias do treino/validação (lesão + vizinhas +
#                  amostra de vazias). Use para comparar com o teste centralizado.
# "completo"     = TODAS as fatias do volume anatômico. Mais rigoroso e realista,
#                  mas os valores tendem a ser menores. Reporte qual usou.


torch.backends.cudnn.benchmark = False


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# 2. IoU volumétrico (o Dice já vem do task.py)
# ============================================================
def calculate_volume_metrics(pred_volume, gt_volume, smooth=1e-6):
    pred = pred_volume.astype(bool)
    gt = gt_volume.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()

    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)

    return (
        float(dice),
        float(iou),
        float(precision),
        float(recall),
    )

# ============================================================
# 3. AVALIAÇÃO DE UM DATASET
# ============================================================
def avaliar_cliente(nome, model, device, modo_fatias):
    """Devolve (dices, ious) — uma entrada por volume de teste deste dataset."""
    data_dir = os.path.join(BASE_DIR, nome)
    pasta_imagens = os.path.join(data_dir, "images")
    pasta_mascaras = os.path.join(data_dir, "segmentations", "expert")
    arquivo_csv = os.path.join(data_dir, "train_test_splits.csv")
    arquivo_csv_stats = os.path.join(data_dir, "volume_stats.csv")

    # --- pacientes do TESTE (coluna nunca usada no treino) ---
    df_splits = pd.read_csv(arquivo_csv)
    pacientes_test = df_splits["test_split"].dropna().astype(str).str.strip().tolist()

    # train=False -> seleciona a fase 0001, igual ao treino
    img_test, mask_test = obter_caminhos_por_lista(
        pacientes_test, pasta_imagens, pasta_mascaras, train=False
    )

    print(f"\n[{nome}] pacientes de teste: {len(pacientes_test)} | volumes: {len(img_test)}")
    if not img_test:
        print(f"[{nome}] AVISO: nenhum volume encontrado. Pulando.")
        return [], []

    set_seed(42)  # antes de construir o dataset (a amostragem de fatias vazias usa RNG)
    test_dataset = MamaMia3DOnTheFlyDataset(
        img_test, mask_test, stats_csv=arquivo_csv_stats
    )

    if modo_fatias == "completo":
        # Reconstrói o mapa para incluir TODAS as fatias de cada volume
        test_dataset.slice_map = [
            (v, z)
            for v in range(len(test_dataset.proxies))
            for z in range(test_dataset.proxies[v][0].shape[2])
        ]

    print(f"[{nome}] fatias de teste ({modo_fatias}): {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    # --- streaming volumétrico: agrupa as fatias por paciente ---
    dices = []
    ious = []
    precisions = []
    recalls = []
    current_vol_id = None
    pred_slices, gt_slices = {}, {}

    def fecha_volume():
        if not pred_slices:
            return
        ordem = sorted(pred_slices.keys())
        pv = np.stack([pred_slices[z] for z in ordem])
        gv = np.stack([gt_slices[z] for z in ordem])
        dice, iou, precision, recall = calculate_volume_metrics(pv, gv)

        dices.append(dice)
        ious.append(iou)
        precisions.append(precision)
        recalls.append(recall)
        pred_slices.clear()
        gt_slices.clear()

    with torch.no_grad():
        for images, masks, vol_idxs, z_idxs in tqdm(test_loader, desc=f"{nome}"):
            images = images.to(device, memory_format=torch.channels_last)
            masks = masks.to(device)

            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
            masks_np = masks.cpu().numpy()
            vol_np = vol_idxs.numpy()
            z_np = z_idxs.numpy()

            for b in range(len(vol_np)):
                vol_id, z_id = int(vol_np[b]), int(z_np[b])
                if current_vol_id is None:
                    current_vol_id = vol_id
                if vol_id != current_vol_id:      # trocou de paciente
                    fecha_volume()
                    current_vol_id = vol_id
                pred_slices[z_id] = preds[b, 0]
                gt_slices[z_id] = masks_np[b, 0]

            del images, masks, outputs
            gc.collect()

        fecha_volume()  # último paciente do buffer

    return dices, ious, precisions, recalls


# ============================================================
# 4. MAIN
# ============================================================
def main():
    set_seed(42)
    device = torch.device("cpu")

    for caminho_modelo in MODELO_PATH:

        # ========================================================
        # CARREGA O MODELO
        # ========================================================
        model = Net(in_channels=1, out_channels=1)

        state = torch.load(
            caminho_modelo,
            map_location=device,
            weights_only=True
        )

        model.load_state_dict(state)
        model = model.to(memory_format=torch.channels_last).to(device)
        model.eval()

        print(f"\nModelo carregado: {caminho_modelo}")

        resultados = []
        por_volume = []

        # ========================================================
        # AVALIA NOS DOIS MODOS
        # ========================================================
        for modo in ["selecionadas", "completo"]:

            print("\n" + "=" * 70)
            print(f"AVALIAÇÃO - FATIAS {modo.upper()}")
            print("=" * 70)

            for nome in CLIENTES:

                dices, ious, precisions, recalls = avaliar_cliente(
                    nome,
                    model,
                    device,
                    modo
                )

                if len(dices) == 0:
                    continue

                resultados.append({
                    "Modo_Fatias": modo,
                    "Dataset": nome,
                    "Dice": np.mean(dices),
                    "Dice_Std": np.std(dices),
                    "IoU": np.mean(ious),
                    "IoU_Std": np.std(ious),
                    "Precision": np.mean(precisions),
                    "Precision_Std": np.std(precisions),
                    "Recall": np.mean(recalls),
                    "Recall_Std": np.std(recalls),
                    "Volumes": len(dices)
                })

                print(
                    f"{nome:8}"
                    f" Dice={np.mean(dices):.4f}±{np.std(dices):.4f}"
                    f" IoU={np.mean(ious):.4f}±{np.std(ious):.4f}"
                    f" Precision={np.mean(precisions):.4f}±{np.std(precisions):.4f}"
                    f" Recall={np.mean(recalls):.4f}±{np.std(recalls):.4f}"
                )

                for idx, (d, i, p, r) in enumerate(
                    zip(dices, ious, precisions, recalls)
                ):
                    por_volume.append({
                        "Modo_Fatias": modo,
                        "Dataset": nome,
                        "Volume": idx,
                        "Dice": d,
                        "IoU": i,
                        "Precision": p,
                        "Recall": r,
                    })

        # ========================================================
        # NOME DOS CSVs BASEADO NO MODELO
        # ========================================================
        nome_modelo = os.path.splitext(os.path.basename(caminho_modelo))[0]

        resumo_csv = f"{nome_modelo}_test_results.csv"
        volumes_csv = f"{nome_modelo}_test_results_por_volume.csv"

        # ========================================================
        # SALVA CSVs
        # ========================================================
        df = pd.DataFrame(resultados)
        dfv = pd.DataFrame(por_volume)

        df.to_csv(resumo_csv, index=False)
        dfv.to_csv(volumes_csv, index=False)

        print("\n" + "=" * 70)
        print(f"RESULTADOS DO MODELO: {nome_modelo}")
        print("=" * 70)
        print(df)

        print(f"\nResumo salvo em: {resumo_csv}")
        print(f"Resultados por volume: {volumes_csv}")

if __name__ == "__main__":
    main()
