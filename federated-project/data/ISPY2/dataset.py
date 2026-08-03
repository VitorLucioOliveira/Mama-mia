"""MAMA-MIA federado: Dataset dinâmico (pré-processamento on-the-fly de NIfTI).

Este arquivo contém APENAS a classe do dataset. Modelo, loss, métricas, load_data,
train e test ficam no task.py (fonte única, para não duplicar).

Retorna (img, mask, vol_idx, z_idx) — os índices permitem remontar o Dice 3D.
"""

import numpy as np
import cv2
import nibabel as nib
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset


class MamaMia3DOnTheFlyDataset(Dataset):
    """Leitura preguiçosa de volumes NIfTI, fatia a fatia."""

    def __init__(self, volume_paths, mask_paths, stats_csv, target_size=(256, 256),
                 only_tumor_slices=False, include_empty_ratio=2):
        self.target_size = target_size
        self.slice_map = []
        self.proxies = []
        self.volume_stats = []

        # -----------------------------------------------------------------
        # CSV de estatísticas (média/desvio por volume).
        # As chaves do CSV terminam em .nii.gz; os volumes agora são .nii,
        # então normalizamos tirando o .gz para a busca casar.
        # -----------------------------------------------------------------
        stats_df = pd.read_csv(stats_csv)
        self.stats_dict = {}
        for _, row in stats_df.iterrows():
            chave = str(row["file"]).replace("\\", "/")
            if chave.endswith(".gz"):
                chave = chave[:-3]          # "...nii.gz" -> "...nii"
            self.stats_dict[chave] = (float(row["mean"]), float(row["std"]))

        print(f"A mapear e abrir proxies para {len(volume_paths)} volumes...")

        for v_path, m_path in zip(volume_paths, mask_paths):
            try:
                relative_path = "/".join(Path(v_path).as_posix().split("/")[-3:])

                if relative_path not in self.stats_dict:
                    raise ValueError(f"Volume não encontrado no CSV: {relative_path}")

                img_nib = nib.load(v_path)
                mask_nib = nib.load(m_path)

                mean, std = self.stats_dict[relative_path]
                if std < 1e-8:
                    std = 1.0

                current_vol_idx = len(self.proxies)
                self.proxies.append((img_nib, mask_nib))
                self.volume_stats.append((mean, std))

                z_slices = img_nib.shape[2]

                if only_tumor_slices:
                    for z in range(z_slices):
                        mask_slice = np.asarray(mask_nib.dataobj[:, :, z])
                        if np.any(mask_slice):
                            self.slice_map.append((current_vol_idx, z))
                else:
                    lesion_slices = []
                    empty_slices = []
                    for z in range(z_slices):
                        mask_slice = np.asarray(mask_nib.dataobj[:, :, z])
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
                                    if 0 <= zz < z_slices and zz not in lesion_slices:
                                        candidate_empty.add(zz)

                        candidate_empty = list(candidate_empty)
                        n_empty = min(len(candidate_empty),
                                      len(lesion_slices) * include_empty_ratio)
                        if n_empty > 0:
                            chosen_empty = np.random.choice(
                                candidate_empty, size=n_empty, replace=False)
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

        img_slice = np.asarray(img_proxy.dataobj[:, :, z_idx], dtype=np.float32)
        mask_slice = np.asarray(mask_proxy.dataobj[:, :, z_idx], dtype=np.float32)
        mask_slice = (mask_slice > 0).astype(np.float32)

        mean, std = self.volume_stats[vol_idx]
        img_slice = (img_slice - mean) / std

        img_resized = cv2.resize(img_slice, self.target_size,
                                 interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask_slice, self.target_size,
                                  interpolation=cv2.INTER_NEAREST)

        img_tensor = torch.from_numpy(img_resized).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_resized).unsqueeze(0)

        return img_tensor, mask_tensor, vol_idx, z_idx