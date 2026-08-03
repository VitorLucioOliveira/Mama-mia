import os
import random
import gc
import time           
import psutil        
import numpy as np
import pandas as pd 
import nibabel as nib
import cv2
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path


# ============================================================
# Configurações Globais Edge/ARM
# ============================================================
torch.backends.cudnn.benchmark = False 
os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
# ============================================================
# Arquitetura Edge: Lightweight U-Net c/ InstanceNorm
# ============================================================
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
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
                    DepthwiseSeparableConv(feature, feature)
                )
            )
            in_channels = feature

        self.bottleneck = nn.Sequential(
            DepthwiseSeparableConv(features[-1], features[-1]*2),
            DepthwiseSeparableConv(features[-1]*2, features[-1]*2)
        )

        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2, bias=False))
            self.ups.append(
                nn.Sequential(
                    DepthwiseSeparableConv(feature*2, feature),
                    DepthwiseSeparableConv(feature, feature)
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
            skip_connection = skip_connections[idx//2]
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)


# ============================================================
# Dataset Dinâmico c/ Cache de Proxies NIfTI
# ============================================================
class MamaMia3DOnTheFlyDataset(Dataset):

    def __init__(self,volume_paths, mask_paths, stats_csv, target_size=(256, 256), only_tumor_slices=False, include_empty_ratio=2):

        self.target_size = target_size
        self.slice_map = []
        self.proxies = []
        self.volume_stats = []

        # =====================================================
        # Carrega CSV de estatísticas
        # =====================================================
        stats_df = pd.read_csv(stats_csv)

        self.stats_dict = {}

        for _, row in stats_df.iterrows():

            self.stats_dict[
                str(row["file"]).replace("\\", "/")
            ] = (
                float(row["mean"]),
                float(row["std"])
            )

        print(f"A mapear e abrir proxies para {len(volume_paths)} volumes...")

        for v_path, m_path in zip(volume_paths, mask_paths):

            try:
                # =====================================================
                # Recupera média/desvio do CSV
                # =====================================================

                relative_path = "/".join(
                    Path(v_path).as_posix().split("/")[-3:]
                )

                if relative_path not in self.stats_dict:

                    raise ValueError(
                        f"Volume não encontrado no CSV: {relative_path}"
                    )
                   
                img_nib = nib.load(v_path)
                mask_nib = nib.load(m_path)

                mean, std = self.stats_dict[relative_path]

                if std < 1e-8:
                    std = 1.0

                # =====================================================
                # Adiciona às estruturas do dataset
                # =====================================================

                current_vol_idx = len(self.proxies)

                self.proxies.append((img_nib, mask_nib))
                self.volume_stats.append((mean, std))

                # =====================================================
                # Mapeamento das slices
                # =====================================================

                z_slices = img_nib.shape[2]

                if only_tumor_slices:

                    for z in range(z_slices):

                        mask_slice = np.asarray(
                            mask_nib.dataobj[:, :, z]
                        )

                        if np.any(mask_slice):
                            self.slice_map.append((current_vol_idx, z))

                else:

                    lesion_slices = []
                    empty_slices = []

                    for z in range(z_slices):

                        mask_slice = np.asarray(
                            mask_nib.dataobj[:, :, z]
                        )

                        if np.any(mask_slice):
                            lesion_slices.append(z)
                        else:
                            empty_slices.append(z)

                    selected_slices = set(lesion_slices)

                    for z in lesion_slices:

                        for offset in (-1, 1):

                            zz = z + offset

                            if 0 <= zz < z_slices:
                                selected_slices.add(zz)

                    if lesion_slices:

                        candidate_empty = set()

                        max_offset = 10

                        for z in lesion_slices:

                            for offset in range(1, max_offset + 1):

                                for sign in (-1, 1):

                                    zz = z + sign * offset

                                    if (
                                        0 <= zz < z_slices
                                        and zz not in lesion_slices
                                    ):
                                        candidate_empty.add(zz)

                        candidate_empty = list(candidate_empty)

                        n_empty = min(
                            len(candidate_empty),
                            len(lesion_slices) * include_empty_ratio
                        )

                        if n_empty > 0:

                            chosen_empty = np.random.choice(
                                candidate_empty,
                                size=n_empty,
                                replace=False
                            )

                            selected_slices.update(chosen_empty)
                            
                    elif not lesion_slices:

                        selected_slices.update(empty_slices)

                    for z in sorted(selected_slices):
                        self.slice_map.append((current_vol_idx, z))

            except Exception as e:

                print(f"Erro ao ler cabeçalho de {v_path}: {e}")

    def __len__(self):
        return len(self.slice_map)

    def __getitem__(self, idx):

        vol_idx, z_idx = self.slice_map[idx]

        img_proxy, mask_proxy = self.proxies[vol_idx]

        img_slice = np.asarray(
            img_proxy.dataobj[:, :, z_idx],
            dtype=np.float32
        )

        mask_slice = np.asarray(
            mask_proxy.dataobj[:, :, z_idx],
            dtype=np.float32
        )

        mask_slice = (mask_slice > 0).astype(np.float32)

        mean, std = self.volume_stats[vol_idx]

        img_slice = (img_slice - mean) / std

        img_resized = cv2.resize(
            img_slice,
            self.target_size,
            interpolation=cv2.INTER_LINEAR
        )

        mask_resized = cv2.resize(
            mask_slice,
            self.target_size,
            interpolation=cv2.INTER_NEAREST
        )

        img_tensor = torch.from_numpy(img_resized).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_resized).unsqueeze(0)
        

        return img_tensor, mask_tensor, vol_idx, z_idx

# ============================================================
# 4. Loss e Métrica Real (Dice)
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
        dice_loss = 1 - (2. * intersection + self.smooth) / (union + self.smooth)
        return (self.bce_weight * bce_loss) + (self.dice_weight * dice_loss.mean())

def calculate_real_dice(logits, targets, smooth=1.0):
    preds = (logits > 0.0).float()
    
    intersection = (preds * targets).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean().item()

def calculate_volume_dice(pred_volume, gt_volume, smooth=1.0):

    pred_volume = pred_volume.astype(np.float32)
    gt_volume = gt_volume.astype(np.float32)

    intersection = (pred_volume * gt_volume).sum()

    dice = (
        2.0 * intersection + smooth
    ) / (
        pred_volume.sum() +
        gt_volume.sum() +
        smooth
    )

    return float(dice)

# ============================================================
# 5. Lógica de Leitura por CSV e Loop Otimizado
# ============================================================
def obter_caminhos_por_lista(lista_pacientes, pasta_imagens, pasta_mascaras, train):
    """Filtra os pacientes e busca imagens e máscaras usando Pathlib de forma segura."""
    img_paths = []
    mask_paths = []
    
    # Converte as strings de pasta para objetos Path (muito mais fácil de manipular)
    dir_img = Path(pasta_imagens)
    dir_mask = Path(pasta_mascaras)
    
    for patient_id in lista_pacientes:
       
        # 1. Limpa qualquer espaço ou quebra de linha que possa ter vindo do CSV
        patient_id = str(patient_id).strip() 
        pasta_paciente = dir_img / patient_id
        
        # 2. Verifica se a pasta do paciente realmente existe
        if not pasta_paciente.is_dir():
            print(f"viso: Pasta de imagem não encontrada -> {pasta_paciente}")
            continue
            
        # 3. Tenta encontrar a máscara de forma inteligente
        mask_file = dir_mask / f"{patient_id}.nii.gz"
            
        if not mask_file.exists():
            print(f"Aviso: Máscara não encontrada para -> {patient_id}")
            continue
            
        # 5. Pega todas as imagens .nii.gz dentro da pasta do paciente
        if not train:
            fases = [
                f for f in pasta_paciente.glob("*.nii.gz")
                if "0001" in  f.name
            ]
        else:
            fases = [
                f for f in pasta_paciente.glob("*.nii.gz")
                if "0000" not in  f.name
            ]
            
        if not fases:
            print(f"Aviso: A pasta {patient_id} está vazia (sem .nii.gz)")
            continue
            
        # 6. Adiciona as fases emparelhadas com a máscara (sem fase pre)
        for img_file in fases:
            
            img_paths.append(str(img_file))
            mask_paths.append(str(mask_file))
            
    return img_paths, mask_paths


# ============================================================
# 6. Checkpointing (retomada após falha de energia / interrupção)
# ============================================================
CHECKPOINT_PATH = "checkpoint_ultimo.pth"
BEST_MODEL_PATH = "best_lightweight_unet_arm.pth"


def salvar_checkpoint(path, epoch, model, optimizer, best_val_dice, epochs_no_improve):
    """Salva um snapshot completo do estado de treino, incluindo os estados
    dos geradores de números aleatórios, para que o treino possa ser retomado
    exatamente de onde parou em caso de falha de energia ou interrupção."""

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_dice": best_val_dice,
        "epochs_no_improve": epochs_no_improve,
        "random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
    }

    # Escreve primeiro num arquivo temporário e só então substitui o
    # checkpoint definitivo, evitando corromper o arquivo caso a energia
    # falhe justamente durante a escrita.
    tmp_path = path + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)


def carregar_checkpoint(path, model, optimizer, device):
    """Carrega um checkpoint existente, se houver, e retorna a época em que
    o treino deve ser retomado (a próxima após a última salva)."""

    if not os.path.exists(path):
        print("\n🆕 Nenhum checkpoint encontrado. Iniciando treinamento do zero.")
        return 0, 0.0, 0

    print(f"\n🔄 Checkpoint encontrado em '{path}'. Retomando treinamento...")
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    random.setstate(checkpoint["random_state"])
    np.random.set_state(checkpoint["numpy_random_state"])
    torch.set_rng_state(checkpoint["torch_random_state"])

    start_epoch = checkpoint["epoch"] + 1
    best_val_dice = checkpoint["best_val_dice"]
    epochs_no_improve = checkpoint["epochs_no_improve"]

    print(f"   -> Retomando a partir da época {start_epoch + 1}")
    print(f"   -> Melhor Dice registrado até então: {best_val_dice:.4f}")
    print(f"   -> Épocas sem melhoria: {epochs_no_improve}")

    return start_epoch, best_val_dice, epochs_no_improve


def train_lightweight_unet():
    
    set_seed(42)
    device = torch.device('cpu') 
    
    # CONFIGS
    epochs = 50
    batch_size = 2
    accumulation_steps = 4
    patience = 15  
    
    # Caminho dos diretorios
    pasta_imagens = "images"
    pasta_mascaras = "segmentations/expert"
    arquivo_csv = "train_test_splits.csv"
    arquivo_csv_stats ="volume_stats.csv"
    
    # Arquivo onde os logs serão salvos
    arquivo_log = "training_metrics_log.csv"

    # LEITURA DO CSV E SPLIT 80/20
    print(f"Lendo listas de pacientes do arquivo: {arquivo_csv}...")
    df_splits = pd.read_csv(arquivo_csv)
    pool_treino = df_splits['train_split'].dropna().astype(str).str.strip().tolist()

    pacientes_train, pacientes_val = train_test_split(pool_treino, test_size=0.2, random_state=42, shuffle=True)
    
    print(f"Total no Pool de Treino do CSV: {len(pool_treino)}")
    print(f" -> Separados para Treinar (80%): {len(pacientes_train)}")
    print(f" -> Separados para Validar (20%): {len(pacientes_val)}")

    img_train, mask_train = obter_caminhos_por_lista(
        pacientes_train, 
        pasta_imagens, 
        pasta_mascaras,
        train=True
    )
    
    img_val, mask_val = obter_caminhos_por_lista(
        pacientes_val, 
        pasta_imagens, 
        pasta_mascaras, 
        train=True
    )

    print(f"Fases (Volumes) carregadas: Treino ({len(img_train)}), Validação ({len(img_val)})")

    train_dataset = MamaMia3DOnTheFlyDataset(
        img_train, 
        mask_train,
        stats_csv=arquivo_csv_stats
    )
    
    val_dataset = MamaMia3DOnTheFlyDataset(
        img_val, 
        mask_val,
        stats_csv= arquivo_csv_stats
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False, persistent_workers=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, persistent_workers=False)
    
    model = LightweightUNet(in_channels=1, out_channels=1)
    model = model.to(memory_format=torch.channels_last).to(device)
    
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # =========================================================
    # Tenta retomar de um checkpoint anterior, se existir
    # =========================================================
    start_epoch, best_val_dice, epochs_no_improve = carregar_checkpoint(
        CHECKPOINT_PATH, model, optimizer, device
    )

    # Cria o cabeçalho do log apenas se o arquivo ainda não existir
    # (se estivermos retomando, o log já existe e deve ser preservado)
    if not os.path.exists(arquivo_log):
        with open(arquivo_log, "w") as f:
            f.write(
                "Epoch,"
                "Train_Loss,"
                "Train_Dice,"
                "Val_Loss,"
                "Val_Dice,"
                "Duration_sec,"
                "CPU_Time_sec,"
                "Peak_RAM_MB\n"
            )

    print("\n🚀 Iniciando Treinamento FP32 + Channels Last no ARM CPU...")
    
    process = psutil.Process(os.getpid())

    
    for epoch in range(start_epoch, epochs):
        
        # Inicia o cronômetro da época
        epoch_start_time = time.perf_counter()

        cpu_start = process.cpu_times()

        peak_ram = process.memory_info().rss / (1024 * 1024)

        
        # ============== TREINAMENTO ==============
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        optimizer.zero_grad() 

        for i, (images, masks, _, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} [Treino]")):
            images = images.to(device, memory_format=torch.channels_last)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            
            loss = loss / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * accumulation_steps
            train_dice += calculate_real_dice(outputs, masks)
            
            # Monitoramento 
            if i % 20 == 0:
                current_ram = process.memory_info().rss / (1024 * 1024)
                peak_ram = max(peak_ram, current_ram)

            
            del images, masks, outputs, loss

        media_train_loss = train_loss / len(train_loader)
        media_train_dice = train_dice / len(train_loader)
    
       # ============== VALIDAÇÃO (Otimizada com Streaming 3D) ==============
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            volume_dices = []
            
            # Variáveis para rastrear o paciente atual em tempo real
            current_vol_id = None
            current_pred_slices = {}
            current_gt_slices = {}
            
            for i, (images, masks, vol_idxs, z_idxs) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1} [Valid]")):
                images = images.to(device, memory_format=torch.channels_last)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
                masks_np = masks.cpu().numpy()
                vol_idxs_np = vol_idxs.numpy()
                z_idxs_np = z_idxs.numpy()
                
                for b in range(len(vol_idxs_np)):
                    vol_id = int(vol_idxs_np[b])
                    z_id = int(z_idxs_np[b])

                    # 1. Inicializa o rastreador no primeiro batch
                    if current_vol_id is None:
                        current_vol_id = vol_id

                    # 2. GATILHO: Mudou de paciente/fase? Calcula o 3D e limpa a RAM!
                    if vol_id != current_vol_id:
                        # Empilha e calcula o Dice do paciente anterior
                        ordered_slices = sorted(current_pred_slices.keys())
                        pred_volume = np.stack([current_pred_slices[z] for z in ordered_slices])
                        gt_volume = np.stack([current_gt_slices[z] for z in ordered_slices])
                        
                        dice = calculate_volume_dice(pred_volume, gt_volume)
                        volume_dices.append(dice)
                        
                        # Limpa os dicionários da RAM e atualiza o rastreador
                        current_pred_slices.clear()
                        current_gt_slices.clear()
                        current_vol_id = vol_id

                    # 3. Guarda a fatia atual na memória temporária
                    current_pred_slices[z_id] = preds[b, 0]
                    current_gt_slices[z_id] = masks_np[b, 0]
                
                # Monitoramento
                if i % 20 == 0:
                    current_ram = process.memory_info().rss / (1024 * 1024)
                    peak_ram = max(peak_ram, current_ram)

                del images, masks, outputs, loss
            
            # GATILHO FINAL: Processa o último paciente que ficou preso no buffer após o loop acabar
            if current_pred_slices:
                ordered_slices = sorted(current_pred_slices.keys())
                pred_volume = np.stack([current_pred_slices[z] for z in ordered_slices])
                gt_volume = np.stack([current_gt_slices[z] for z in ordered_slices])
                
                dice = calculate_volume_dice(pred_volume, gt_volume)
                volume_dices.append(dice)
                
        media_val_loss = val_loss / len(val_loader)
        media_val_volume_dice = np.mean(volume_dices)
        
        gc.collect() 
        
        # ============== COLETA DE MÉTRICAS DA ÉPOCA ==============
        epoch_duration = time.perf_counter() - epoch_start_time

        cpu_end = process.cpu_times()

        cpu_time_used = (
            (cpu_end.user + cpu_end.system)
            -
            (cpu_start.user + cpu_start.system)
        )


        # Salva as métricas no CSV
        with open(arquivo_log, "a") as f:
            f.write(
                f"{epoch+1},"
                f"{media_train_loss:.4f},"
                f"{media_train_dice:.4f},"
                f"{media_val_loss:.4f},"
                f"{media_val_volume_dice:.4f},"
                f"{epoch_duration:.2f},"
                f"{cpu_time_used:.2f},"
                f"{peak_ram:.2f}\n"
            )

        # Exibição no Terminal
        print(f" Época {epoch+1}/{epochs} finalizada em {epoch_duration:.1f}s")
        print(f"    CPU Time: {cpu_time_used:.1f}s | RAM Pico: {peak_ram:.1f} MB")
        print(f"    Train Loss: {media_train_loss:.4f} | Train Dice: {media_train_dice:.4f}")
        print(f"    Val Loss: {media_val_loss:.4f} | Val Dice: { media_val_volume_dice:.4f}")
        
        # SÓ SALVA O MELHOR MODELO E CONTROLA O EARLY STOPPING
        if  media_val_volume_dice > best_val_dice:
            print(f" Melhor Dice alcançado! ({best_val_dice:.4f} -> { media_val_volume_dice:.4f}). Guardando modelo...")
            best_val_dice =  media_val_volume_dice
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            epochs_no_improve = 0  # Reseta a paciência pois houve melhoria
        else:
            epochs_no_improve += 1
            print(f" Sem melhoria no Dice. Paciência: {epochs_no_improve}/{patience}")

        # =========================================================
        # Salva o checkpoint de retomada AO FIM DE CADA ÉPOCA.
        # Isso garante que, em caso de falha de energia, o treino
        # possa ser retomado exatamente da próxima época, com o
        # otimizador e os geradores aleatórios no mesmo estado.
        # =========================================================
        salvar_checkpoint(
            CHECKPOINT_PATH,
            epoch,
            model,
            optimizer,
            best_val_dice,
            epochs_no_improve,
        )
        print(f"    💾 Checkpoint de retomada salvo em '{CHECKPOINT_PATH}' (época {epoch+1}).")

        # Verifica se atingiu o limite de paciência
        if epochs_no_improve >= patience:
            print(f"\n Early Stopping acionado! O treinamento foi interrompido na época {epoch+1}.")
            break  # Quebra o loop 'for' das épocas

    print("\n✅ Treinamento finalizado.")

if __name__ == "__main__":
    train_lightweight_unet()
