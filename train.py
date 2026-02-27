import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from dataset import DukeDataset
from model import SimpleUNet
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------- LOSS FUNCTIONS ---------------- #

def dice_loss(logits, masks, smooth=1e-8):

    probs = torch.sigmoid(logits)

    probs = probs.view(probs.size(0), -1)
    masks = masks.view(masks.size(0), -1)

    intersection = (probs * masks).sum(dim=1)
    union = probs.sum(dim=1) + masks.sum(dim=1)

    dice = (2 * intersection + smooth) / (union + smooth)

    return 1 - dice.mean()


# Dice métrico (soft)
def calculate_dice(logits, masks, smooth=1e-8):

    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    preds = preds.view(preds.size(0), -1)
    masks = masks.view(masks.size(0), -1)

    intersection = (preds * masks).sum(dim=1)
    union = preds.sum(dim=1) + masks.sum(dim=1)

    dice = (2 * intersection + smooth) / (union + smooth)

    return dice.mean()


# IoU / Jaccard
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
    
    #------------- Configs Iniciais -------------------#

    CONFIG = {
        'seed': 42,
        'batch_size':4,
        'lr': 0.0001,
        'num_epochs':50
    }

    set_seed(CONFIG['seed'])

    # Pega a lista de treino e test disponiveis
    split_df = pd.read_csv("dados_processados_2d/train_test_splits.csv")

    lista_treino = split_df['train_split'].dropna().tolist()
    lista_test = split_df['test_split'].dropna().tolist()

    dataset_treino = DukeDataset("dados_processados_2d", lista_treino)
    dataset_test = DukeDataset("dados_processados_2d", lista_test)

    train_loader = DataLoader(dataset_treino, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(dataset_test, batch_size=CONFIG['batch_size'], shuffle=False)

    model = SimpleUNet(n_channels=1, n_classes=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # BCE com peso para classe positiva
    pos_weight = torch.tensor([5.0]).to(device)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def combined_loss(logits, masks):
        return bce_loss(logits, masks) + dice_loss(logits, masks)

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        patience=5,
        factor=0.5
    )

    num_epochs = CONFIG['num_epochs']
    melhor_val_dice = 0.0

    # Elementos para graficos
    historico_loss_treino, historico_loss_val = [],[]
    historico_dice_treino , historico_dice_val = [],[]
    historico_iou_treino , historico_iou_val = [],[]

    #------------- Loop de Treinamento --------------#

    for i in range(num_epochs):

        total_loss_treino = total_loss_val = 0.0
        total_dice_treino = total_dice_val = 0.0
        total_iou_treino = total_iou_val = 0.0

        total_amostras_treino = 0
        total_amostras_val = 0

        # -------- TREINO -------- #

        model.train()

        for train_images, train_masks in tqdm(train_loader, desc=f"Treino Época {i+1}"):

            train_images = train_images.to(device)
            train_masks = train_masks.to(device)

            optimizer.zero_grad()

            train_previsoes = model(train_images)

            train_loss = combined_loss(train_previsoes, train_masks)

            train_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            train_dice = calculate_dice(train_previsoes, train_masks)
            train_iou = calculate_iou(train_previsoes, train_masks)

            train_size = train_images.size(0)

            total_amostras_treino += train_size

            total_loss_treino += train_loss.item() * train_size
            total_dice_treino += train_dice.item() * train_size
            total_iou_treino += train_iou.item() * train_size


        # -------- VALIDAÇÃO -------- #

        model.eval()

        with torch.no_grad():

            for val_images, val_masks in tqdm(val_loader, desc=f"Teste Época {i+1}"):

                val_images = val_images.to(device)
                val_masks = val_masks.to(device)

                val_previsoes = model(val_images)

                val_loss = combined_loss(val_previsoes, val_masks)

                val_dice = calculate_dice(val_previsoes, val_masks)
                val_iou = calculate_iou(val_previsoes, val_masks)

                val_size = val_images.size(0)

                total_amostras_val += val_size

                total_loss_val += val_loss.item() * val_size
                total_dice_val += val_dice.item() * val_size
                total_iou_val += val_iou.item() * val_size


        # -------- MÉDIAS -------- #

        media_train_loss = total_loss_treino / total_amostras_treino
        media_train_dice = total_dice_treino / total_amostras_treino
        media_train_iou = total_iou_treino / total_amostras_treino

        media_val_loss = total_loss_val / total_amostras_val
        media_val_dice = total_dice_val / total_amostras_val
        media_val_iou = total_iou_val / total_amostras_val

        # Scheduler
        scheduler.step(media_val_dice)

        # Histórico

        historico_loss_treino.append(media_train_loss)
        historico_loss_val.append(media_val_loss)

        historico_dice_treino.append(media_train_dice)
        historico_dice_val.append(media_val_dice)

        historico_iou_treino.append(media_train_iou)
        historico_iou_val.append(media_val_iou)

        # Salvar melhor modelo

        if media_val_dice > melhor_val_dice:

            melhor_val_dice = media_val_dice
            torch.save(model.state_dict(), "melhor_unet.pth")
            print("Novo recorde! Modelo salvo.")

        print(
            f"Época [{i+1}/{num_epochs}] "
            f"| Loss Treino: {media_train_loss:.4f} "
            f"| Loss Val: {media_val_loss:.4f} "
            f"| Dice Treino: {media_train_dice:.4f} "
            f"| Dice Val: {media_val_dice:.4f} "
            f"| IoU Val: {media_val_iou:.4f}"
        )


    #------------- Plotando os Gráficos --------------#

    print("\nGerando gráficos de treinamento...")

    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # LOSS
    axs[0].plot(historico_loss_treino, label='Treino')
    axs[0].plot(historico_loss_val, label='Validação')
    axs[0].set_title('Loss')
    axs[0].set_xlabel('Épocas')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid(True)

    # DICE
    axs[1].plot(historico_dice_treino, label='Treino')
    axs[1].plot(historico_dice_val, label='Validação')
    axs[1].set_title('Dice Score')
    axs[1].set_xlabel('Épocas')
    axs[1].set_ylabel('Dice')
    axs[1].legend()
    axs[1].grid(True)

    # IOU
    axs[2].plot(historico_iou_treino, label='Treino')
    axs[2].plot(historico_iou_val, label='Validação')
    axs[2].set_title('IoU (Jaccard Index)')
    axs[2].set_xlabel('Épocas')
    axs[2].set_ylabel('IoU')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()

    plt.savefig("graficos_treinamento.png", dpi=300)

    print("Gráficos salvos em alta resolução!")