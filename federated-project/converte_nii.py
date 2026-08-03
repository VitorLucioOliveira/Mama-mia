"""Converte todos os .nii.gz para .nii descomprimido e APAGA os .nii.gz (roda UMA vez, no host).

Motivo: gzip não permite acesso aleatório. Ler fatia por fatia de um .nii.gz
re-descomprime o arquivo inteiro a cada acesso, o que trava o treino federado.
Arquivos .nii descomprimidos são mapeados em memória e o acesso vira instantâneo.

ATENÇÃO: este script é destrutivo — ele remove os .nii.gz originais. O original
só é apagado DEPOIS de confirmar que o .nii foi gravado e abre corretamente.
Garanta que você tem um backup (ou consegue baixar o dataset de novo) antes de rodar.
"""

import glob
import os
import nibabel as nib

BASE = "data"  # pasta que contém DUKE, ISPY1, ISPY2, NACT

arquivos = glob.glob(os.path.join(BASE, "**", "*.nii.gz"), recursive=True)
print(f"{len(arquivos)} arquivos .nii.gz encontrados.")

convertidos = 0
erros = 0

for i, path in enumerate(arquivos, 1):
    destino = path[:-3]  # remove o '.gz' -> termina em .nii
    try:
        # 1. Converte (se o .nii ainda não existe)
        if not os.path.exists(destino):
            nib.save(nib.load(path), destino)

        # 2. Verifica que o .nii ficou válido ANTES de apagar o original
        vol = nib.load(destino)
        _ = vol.shape                       # força a leitura do cabeçalho
        if os.path.getsize(destino) == 0:
            raise ValueError("arquivo .nii ficou vazio")

        # 3. Só então remove o .nii.gz original
        os.remove(path)
        convertidos += 1

    except Exception as e:
        erros += 1
        print(f"  ERRO em {path}: {e} (o .nii.gz foi MANTIDO)")

    if i % 50 == 0 or i == len(arquivos):
        print(f"  {i}/{len(arquivos)} processados")

print(f"\nConcluído. {convertidos} convertidos (originais removidos), "
      f"{erros} com erro (originais mantidos).")
