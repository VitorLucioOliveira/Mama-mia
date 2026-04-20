import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from dataset import Dataset  # Certifique-se que sua classe no dataset.py tem este nome
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
import os

# Otimizações importantes para WSL + pouca RAM
torch.backends.cudnn.benchmark = True          # acelera convoluções
torch.backends.cuda.matmul.allow_tf32 = True   # menos VRAM
torch.backends.cudnn.allow_tf32 = True         # menos VRAM

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ---------------- Métricas e Losses ---------------- #

def dice_loss(logits, masks, smooth=1e-8):
    probs = torch.sigmoid(logits)
    probs = probs.view(probs.size(0), -1)
    masks = masks.view(masks.size(0), -1)
    intersection = (probs * masks).sum(dim=1)
    union = probs.sum(dim=1) + masks.sum(dim=1)
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def calculate_dice(logits, masks, smooth=1e-8):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    preds = preds.view(preds.size(0), -1)
    masks = masks.view(masks.size(0), -1)
    intersection = (preds * masks).sum(dim=1)
    union = preds.sum(dim=1) + masks.sum(dim=1)
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice.mean()

def calculate_iou(logits, masks, smooth=1e-8):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    preds = preds.view(preds.size(0), -1)
    masks = masks.view(masks.size(0), -1)
    intersection = (preds * masks).sum(dim=1)
    union = preds.sum(dim=1) + masks.sum(dim=1) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()

if __name__ == "__main__":
    
    #------------- Configurações -------------#
    CONFIG = {
        'seed': 42,
        'batch_size': 32,
        'lr': 0.0001,
        'num_epochs': 10,
        'accum_steps': 2
    }

    set_seed(CONFIG['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    pin_memory = torch.cuda.is_available()

    # Preparação dos Dados
    split_df = pd.read_csv("train_test_splits.csv")
    lista_treino_base = split_df['train_split'].dropna().tolist()
    
    
    # Treinamentos por tipo de xxx
    df_clinico = pd.read_csv("clinical_data.csv")
    coluna_alvo = 'manufacturer' 
    grupos_unicos = df_clinico[coluna_alvo].dropna().unique()
    
    print(f"Grupos encontrados para treinamento: {grupos_unicos}")
    
    for grupo in grupos_unicos:
        print(f"\n{'='*50}")
        print(f"🚀 INICIANDO TREINAMENTO PARA: {grupo}")
        print(f"{'='*50}")
        
        # Limpa o nome para usar nos arquivos (troca espaço por underline)
        grupo_limpo = str(grupo).replace(" ", "_").replace("/", "_")
        
        # Filtra os pacientes que pertencem a este grupo
        pacientes_do_grupo = df_clinico[df_clinico[coluna_alvo] == grupo]['patient_id'].tolist()
        
        # Interseção: Pacientes do grupo que também estão no split de treino
        lista_treino_filtrada = [p for p in lista_treino_base if p in pacientes_do_grupo]
        
        if len(lista_treino_filtrada) == 0:
            print(f"Aviso: Nenhum paciente de treino encontrado para {grupo}. Pulando...")
            continue
            
        # Faz o split de treino/validação especificamente para este grupo
        lista_treino_final, lista_val = train_test_split(
            lista_treino_filtrada, test_size=0.2, random_state=CONFIG['seed']
        )
        
        # Datasets (fase_especifica=None para usar todas as fases DCE)
        dataset_treino = Dataset("dados_processados_2d", lista_treino_final, fase_especifica="0001", is_train=True)
        dataset_val = Dataset("dados_processados_2d", lista_val, fase_especifica="0001", is_train=False)

        train_loader = DataLoader(
            dataset_treino,
            batch_size=CONFIG['batch_size'],
            shuffle=True,
            num_workers=1,            # ← REDUZ RAM (antes era 2)
           pin_memory=pin_memory,
            prefetch_factor=2,        # ← controla pré-carregamento
            persistent_workers=False  # ← LIBERA RAM entre épocas
        )

        val_loader = DataLoader(
            dataset_val,
            batch_size=CONFIG['batch_size'],
            shuffle=False,
            num_workers=1,
           pin_memory=pin_memory,
            prefetch_factor=2,
            persistent_workers=False
        )

        # Inicialização: MobileNet-V2 como Encoder da UNet
        model = smp.Unet(
            encoder_name="mobilenet_v2",
            encoder_weights="imagenet",
            in_channels=1,
            classes=1
        ).to(device)


        # Loss Composta 
        bce_loss = nn.BCEWithLogitsLoss()
        def combined_loss(logits, masks):
            return (0.5 * bce_loss(logits, masks)) + (0.5 * dice_loss(logits, masks))

        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

        # Histórico para os gráficos
        melhor_val_dice = 0.0
        historico = {
            'loss_treino': [], 'loss_val': [],
            'dice_treino': [], 'dice_val': [],
            'iou_treino': [], 'iou_val': [],
            'lr': []
        }

        #------------- Loop de Treinamento --------------#

        for epoch in range(CONFIG['num_epochs']):
            metrics_epoch = {'loss_t': 0, 'dice_t': 0, 'iou_t': 0, 'loss_v': 0, 'dice_v': 0, 'iou_v': 0}
            
            # TREINO
            model.train()
            for i, (images, masks) in enumerate(tqdm(train_loader, desc=f"Época {epoch+1} [Treino]")):
                
                images, masks = images.to(device), masks.to(device)
                
                # zera gradientes só quando começa acumulação
                if i % CONFIG['accum_steps'] == 0:
                    optimizer.zero_grad(set_to_none=True)
                
                # forward em mixed precision
                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = model(images)
                    loss = combined_loss(outputs, masks) / CONFIG['accum_steps']

                # acumula gradientes
                scaler.scale(loss).backward()

                # quando completar N batches → faz update real
                if (i + 1) % CONFIG['accum_steps'] == 0:

                    # unscale antes de clipar
                    scaler.unscale_(optimizer)

                    # gradient clipping correto
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    # step real
                    scaler.step(optimizer)
                    scaler.update()

                batch_size = images.size(0)
                metrics_epoch['loss_t'] += loss.item() * batch_size
                metrics_epoch['dice_t'] += calculate_dice(outputs, masks).item() * batch_size
                metrics_epoch['iou_t'] += calculate_iou(outputs, masks).item() * batch_size

            if (i + 1) % CONFIG['accum_steps'] != 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                        
            # VALIDAÇÃO
            model.eval()
            with torch.no_grad():
                for images, masks in tqdm(val_loader, desc=f"Época {epoch+1} [Val]"):
                    images, masks = images.to(device), masks.to(device)

                    with torch.amp.autocast("cuda", enabled=use_amp):
                        outputs = model(images)
                        loss = combined_loss(outputs, masks)
    
                    batch_size = images.size(0)
                    metrics_epoch['loss_v'] += loss.item() * batch_size
                    metrics_epoch['dice_v'] += calculate_dice(outputs, masks).item() * batch_size
                    metrics_epoch['iou_v'] += calculate_iou(outputs, masks).item() * batch_size

            # Médias da Época
            n_treino, n_val = len(dataset_treino), len(dataset_val)
            historico['loss_treino'].append(metrics_epoch['loss_t'] / n_treino)
            historico['loss_val'].append(metrics_epoch['loss_v'] / n_val)
            historico['dice_treino'].append(metrics_epoch['dice_t'] / n_treino)
            historico['dice_val'].append(metrics_epoch['dice_v'] / n_val)
            historico['iou_treino'].append(metrics_epoch['iou_t'] / n_treino)
            historico['iou_val'].append(metrics_epoch['iou_v'] / n_val)
            historico['lr'].append(optimizer.param_groups[0]['lr'])
            
            torch.cuda.empty_cache() # limpa cache
            
            # Atualiza Scheduler com base no Dice de Validação
            scheduler.step(historico['dice_val'][-1])

            # Salva o melhor modelo
            if historico['dice_val'][-1] > melhor_val_dice:
                melhor_val_dice = historico['dice_val'][-1]
                
                nome_arquivo_modelo = f"mobilenet_modelo_{grupo_limpo}.pth"
                torch.save(model.state_dict(), nome_arquivo_modelo)
                
                print(f"⭐ Novo recorde: Dice {melhor_val_dice:.4f} salvo em {nome_arquivo_modelo}")

       #------------- Plotagem Estilo Jupyter --------------#
        print(f"\n📊 Gerando gráficos finais para {grupo}...")
        fig, axs = plt.subplots(4, 1, figsize=(12, 20))
        epochs_range = range(1, CONFIG['num_epochs'] + 1)
        
        # LOSS
        axs[0].plot(epochs_range, historico['loss_treino'], 'b-o', label='Treino')
        axs[0].plot(epochs_range, historico['loss_val'], 'r-o', label='Validação')
        axs[0].set_title(f'Curva de Perda (Mixed Loss) - {grupo}') # <-- ALTERADO
        axs[0].set_ylabel('Loss')
        
        # DICE
        axs[1].plot(epochs_range, historico['dice_treino'], 'g-s', label='Treino')
        axs[1].plot(epochs_range, historico['dice_val'], 'orange', marker='s', label='Validação')
        axs[1].set_title(f'Evolução do Dice Score - {grupo}') # <-- ALTERADO
        axs[1].set_ylabel('Dice')
        
        # IOU
        axs[2].plot(epochs_range, historico['iou_treino'], 'm-d', label='Treino')
        axs[2].plot(epochs_range, historico['iou_val'], 'c-d', label='Validação')
        axs[2].set_title(f'Evolução do IoU - {grupo}') # <-- ALTERADO
        axs[2].set_ylabel('IoU Score')

        # LEARNING RATE
        axs[3].plot(epochs_range, historico['lr'], color='black', linestyle='--', label='LR')
        axs[3].set_title(f'Taxa de Aprendizagem (Scheduler) - {grupo}') # <-- ALTERADO
        axs[3].set_ylabel('LR'); axs[3].set_yscale('log')

        for ax in axs:
            ax.set_xlabel('Épocas'); ax.legend(); ax.grid(True, linestyle=':', alpha=0.6)
        
        plt.tight_layout()
        
        
        nome_arquivo_grafico = f"resultados_finais_mobilenet_{grupo_limpo}.png"
        plt.savefig(nome_arquivo_grafico, dpi=300)
        print(f"✅ Processo concluído para {grupo}!")