import pandas as pd
import torch
from torch.utils.data import DataLoader
from dataset import Dataset 
from train import calculate_dice, calculate_iou
from tqdm import tqdm 
import numpy as np
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import seaborn as sns
import random
from collections import defaultdict
import os

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ---------------------------------------------------------
# Escolher slices de PACIENTES DIFERENTES para visualização
# ---------------------------------------------------------
def escolher_indices_pacientes_diferentes(dataset, max_amostras=3):
    pacientes_dict = defaultdict(list)
    for i, path in enumerate(dataset.files):
        paciente = path.split("/")[-1].split("_")[0]
        _, mask = dataset[i]
        if mask.sum() > 50:
            pacientes_dict[paciente].append(i)

    pacientes = list(pacientes_dict.keys())
    random.shuffle(pacientes)

    indices = []
    for p in pacientes:
        indices.append(random.choice(pacientes_dict[p]))
        if len(indices) == max_amostras:
            break
    return indices

# ---------------------------------------------------------
# Salvar Predição Individual (Ranking)
# ---------------------------------------------------------
def salvar_predicao_individual(img, mask, pred_binaria, dice, save_path, titulo):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    axes[0].set_title("Original", fontweight='bold')
    axes[0].imshow(img, cmap='gray')
    axes[0].axis('off')

    axes[1].set_title("Ground Truth", fontweight='bold')
    axes[1].imshow(mask, cmap='hot')
    axes[1].axis('off')

    axes[2].set_title(titulo, fontweight='bold', fontsize=10)
    axes[2].imshow(img, cmap='gray')
    axes[2].imshow(pred_binaria, cmap='turbo', alpha=0.5)
    axes[2].text(5, 15, f'Dice: {dice:.3f}', color='yellow', fontsize=10, fontweight='bold', bbox=dict(facecolor='black', alpha=0.6))
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":

    #------------- Configs Iniciais (Otimizado para CPU) -------------------#
    device = torch.device('cpu') 
    pin_memory = False           
    batch_size_teste = 4         # Batch seguro para processador

    PASTA_RESULTADOS = "Testes_manufacture"
    os.makedirs(PASTA_RESULTADOS, exist_ok=True)

    split_df = pd.read_csv('train_test_splits.csv')
    lista_teste_base = split_df["test_split"].dropna().tolist()
    
    df_clinico = pd.read_csv("clinical_data.csv")
    coluna_alvo = 'manufacturer' 
    grupos_unicos = df_clinico[coluna_alvo].dropna().unique()
    
    print(f"Grupos encontrados para teste: {grupos_unicos}")

    # Matrizes para os resultados finais
    resultados_cross = {}
    matriz_dice_heatmap = pd.DataFrame(index=grupos_unicos, columns=grupos_unicos, dtype=float)

    # ==========================================================
    # OTIMIZAÇÃO: PRÉ-CARREGAR DATASETS NA RAM
    # ==========================================================
    print("\n📦 Pré-carregando Datasets na memória...")
    datasets_por_grupo = {}
    loaders_por_grupo = {}
    
    for grupo_teste in grupos_unicos:
        pacientes_do_grupo = df_clinico[df_clinico[coluna_alvo] == grupo_teste]['patient_id'].tolist()
        lista_teste_filtrada = [p for p in lista_teste_base if p in pacientes_do_grupo]
        
        if len(lista_teste_filtrada) > 0:
            ds = Dataset("dados_processados_2d", lista_teste_filtrada, fase_especifica="0001", is_train=False)
            dl = DataLoader(ds, batch_size=batch_size_teste, shuffle=False, num_workers=0, pin_memory=pin_memory)
            datasets_por_grupo[grupo_teste] = ds
            loaders_por_grupo[grupo_teste] = dl
            print(f"  ↳ {grupo_teste}: {len(ds)} imagens prontas.")

    # ==========================================================
    # LOOP EXTERNO: MODELO TREINADO EM CADA FABRICANTE
    # ==========================================================
    for grupo_modelo in grupos_unicos:
        grupo_modelo_limpo = str(grupo_modelo).replace(" ", "_").replace("/", "_")
        nome_modelo = f"mobilenet_modelo_{grupo_modelo_limpo}.pth"

        pasta_modelo = os.path.join(PASTA_RESULTADOS, f"ranking_imagens_{grupo_modelo_limpo}")
        os.makedirs(pasta_modelo, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"🔍 CARREGANDO MODELO TREINADO EM: {grupo_modelo}")
        print(f"{'='*60}")

        model = smp.Unet(encoder_name="mobilenet_v2", encoder_weights="imagenet", in_channels=1, classes=1)
        
        try:
            model.load_state_dict(torch.load(nome_modelo, map_location=device))
            model.to(device)
            model.eval()
        except FileNotFoundError:
            print(f"❌ Modelo {nome_modelo} não encontrado. Pulando...")
            continue

        resultados_cross[grupo_modelo] = {}

        # ======================================================
        # LOOP INTERNO: TESTAR EM TODOS OS FABRICANTES
        # ======================================================
        for grupo_teste in grupos_unicos:
            if grupo_teste not in loaders_por_grupo:
                matriz_dice_heatmap.loc[grupo_modelo, grupo_teste] = np.nan
                continue

            print(f"\n{'-'*40}")
            print(f"🧪 MODELO {grupo_modelo} → TESTE EM {grupo_teste}")
            
            dataset_teste = datasets_por_grupo[grupo_teste]
            test_loader = loaders_por_grupo[grupo_teste]
            
            lista_dice, lista_iou = [], []
            ranking_slices = []   
            indice_global_dataset = 0 

            with torch.no_grad():
                for test_images, test_masks in tqdm(test_loader, desc=f"Avaliando"):
                    test_images = test_images.to(device)
                    test_masks = test_masks.to(device)

                    previsoes = model(test_images)

                    batch_dice = calculate_dice(previsoes, test_masks).item()
                    batch_iou = calculate_iou(previsoes, test_masks).item()
                    lista_dice.append(batch_dice)
                    lista_iou.append(batch_iou)

                    # ---------- RANKING VETORIZADO (Intra-grupo) ----------
                    if grupo_modelo == grupo_teste:  
                        pred_prob = torch.sigmoid(previsoes)
                        pred_binaria = (pred_prob > 0.5).float()
                        
                        preds_flat = pred_binaria.view(pred_binaria.size(0), -1)
                        masks_flat = test_masks.view(test_masks.size(0), -1)
                        intersecao_batch = (preds_flat * masks_flat).sum(dim=1)
                        uniao_batch = preds_flat.sum(dim=1) + masks_flat.sum(dim=1)
                        dices_fatia = (2 * intersecao_batch) / (uniao_batch + 1e-8)

                        bs_atual = test_images.size(0)
                        for b in range(bs_atual):
                            idx_real = indice_global_dataset + b
                            
                            if masks_flat[b].sum() > 0:
                                caminho_arquivo = dataset_teste.files[idx_real]
                                nome_arquivo = os.path.basename(caminho_arquivo)
                                partes = nome_arquivo.split('_')
                                dataset_origem = partes[0] 
                                paciente_id = f"{partes[0]}_{partes[1]}" 

                                ranking_slices.append({
                                    "dice": dices_fatia[b].item(),
                                    "img": test_images[b].cpu().squeeze().numpy(),
                                    "mask": test_masks[b].cpu().squeeze().numpy(),
                                    "pred": pred_binaria[b].cpu().squeeze().numpy(),
                                    "dataset": dataset_origem,
                                    "paciente_id": paciente_id
                                })
                                
                                if len(ranking_slices) > 1000:
                                    ranking_slices.sort(key=lambda x: x["dice"])
                                    ranking_slices = ranking_slices[:150] + ranking_slices[-150:]
                                    
                    indice_global_dataset += test_images.size(0)

            # --- RESULTADOS MÉDIOS ---
            dice_medio = np.mean(lista_dice)
            iou_medio = np.mean(lista_iou)
            print(f"↳ Dice: {dice_medio:.4f} | IoU: {iou_medio:.4f}")

            matriz_dice_heatmap.loc[grupo_modelo, grupo_teste] = dice_medio
            resultados_cross[grupo_modelo][grupo_teste] = {
                'dice_mean': dice_medio, 'dice_std': np.std(lista_dice),
                'iou_mean': iou_medio, 'iou_std': np.std(lista_iou)
            }
            
            # ==================================================
            # SALVAR 100 MELHORES E 100 PIORES
            # ==================================================
            if grupo_modelo == grupo_teste and len(ranking_slices) > 0:
                print("📸 Salvando ranking de fatias...")
                ranking_ordenado = sorted(ranking_slices, key=lambda x: x["dice"])
                worst_100, best_100 = ranking_ordenado[:100], ranking_ordenado[-100:]

                pasta_best = os.path.join(pasta_modelo, "best_100")
                pasta_worst = os.path.join(pasta_modelo, "worst_100")
                os.makedirs(pasta_best, exist_ok=True); os.makedirs(pasta_worst, exist_ok=True)

                for i, item in enumerate(worst_100):
                    salvar_predicao_individual(
                        item["img"], item["mask"], item["pred"], item["dice"],
                        os.path.join(pasta_worst, f"worst_{i:03d}_{item['dataset']}_{item['paciente_id']}_dice_{item['dice']:.3f}.png"),
                        f"Pior ({grupo_modelo}) | {item['dataset']} - {item['paciente_id']}"
                    )
                for i, item in enumerate(best_100[::-1]):
                    salvar_predicao_individual(
                        item["img"], item["mask"], item["pred"], item["dice"],
                        os.path.join(pasta_best, f"best_{i:03d}_{item['dataset']}_{item['paciente_id']}_dice_{item['dice']:.3f}.png"),
                        f"Melhor ({grupo_modelo}) | {item['dataset']} - {item['paciente_id']}"
                    )

            # ==================================================
            # PREDIÇÕES VISUAIS ALEATÓRIAS
            # ==================================================
            if grupo_modelo == grupo_teste:
                indices_escolhidos = escolher_indices_pacientes_diferentes(dataset_teste)
                if len(indices_escolhidos) > 0:
                    amostras = [dataset_teste[i] for i in indices_escolhidos]
                    fig, axes = plt.subplots(len(amostras), 3, figsize=(10, 3 * len(amostras)))
                    if len(amostras) == 1: axes = [axes]

                    titulos = ['Original', 'Ground Truth', f'Predição ({grupo_modelo})']
                    for ax, title in zip(axes[0], titulos): ax.set_title(title, fontsize=12, fontweight='bold')

                    for row, (img, mask) in enumerate(amostras):
                        img_tensor = img.unsqueeze(0).to(device)
                        with torch.no_grad():
                            pred = torch.sigmoid(model(img_tensor)).cpu().squeeze()
                            pred_binaria = (pred > 0.5).float()

                        axes[row][0].imshow(img.squeeze().numpy(), cmap='gray'); axes[row][0].axis('off')
                        axes[row][1].imshow(mask.squeeze().numpy(), cmap='hot'); axes[row][1].axis('off')
                        axes[row][2].imshow(img.squeeze().numpy(), cmap='gray')
                        axes[row][2].imshow(pred_binaria.numpy(), cmap='turbo', alpha=0.5)
                        
                        intersecao = (pred_binaria.numpy() * mask.squeeze().numpy()).sum()
                        uniao = pred_binaria.numpy().sum() + mask.squeeze().numpy().sum()
                        axes[row][2].text(5, 15, f'Dice: {(2 * intersecao) / (uniao + 1e-8):.2f}', color='yellow', fontsize=10, fontweight='bold', bbox=dict(facecolor='black', alpha=0.6))
                        axes[row][2].axis('off')

                    plt.tight_layout()
                    plt.savefig(os.path.join(PASTA_RESULTADOS, f"predicoes_aleatorias_{grupo_modelo_limpo}.png"), dpi=200, bbox_inches='tight')
                    plt.close()

    # ======================================================
    # ARTEFATOS FINAIS: TABELA CSV E HEATMAP
    # ======================================================
    print("\n📊 Gerando artefatos comparativos finais...")
    
    # 1. Tabela CSV
    linhas = []
    for modelo, testes in resultados_cross.items():
        for grupo_teste, v in testes.items():
            linhas.append([f"{modelo} → {grupo_teste}", f"{v['dice_mean']:.4f} ± {v['dice_std']:.4f}", f"{v['iou_mean']:.4f} ± {v['iou_std']:.4f}"])

    df_resultados = pd.DataFrame(linhas, columns=["Treino → Teste", "Dice (mean ± std)", "IoU (mean ± std)"])
    caminho_csv = os.path.join(PASTA_RESULTADOS, "tabela_cross_group.csv")
    df_resultados.to_csv(caminho_csv, index=False)
    
    # 2. Heatmap do Domain Shift
    plt.figure(figsize=(10, 8))
    sns.heatmap(matriz_dice_heatmap.astype(float), annot=True, cmap="YlGnBu", fmt=".3f", 
                cbar_kws={'label': 'Dice Score Médio'}, linewidths=.5, square=True)
    
    plt.title('Matriz de Testes Cruzados (Cross-Testing)\nDomain Shift por Fabricante', fontsize=14, pad=20)
    plt.ylabel('Fabricante do Modelo Treinado (Origem)', fontsize=12, fontweight='bold')
    plt.xlabel('Fabricante dos Dados de Teste (Alvo)', fontsize=12, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    caminho_heatmap = os.path.join(PASTA_RESULTADOS, "heatmap_cross_testing.png")
    plt.savefig(caminho_heatmap, dpi=300)
    plt.close()

    print(f"✅ Concluído! Todos os resultados, rankings e o heatmap foram salvos na pasta '{PASTA_RESULTADOS}'.")