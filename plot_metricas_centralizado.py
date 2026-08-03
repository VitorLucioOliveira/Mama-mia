import os
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

# ============================================================
# CONFIGURAÇÃO
# ============================================================

ARQUIVOS = {
    "DUKE":  "training_metrics_log_duke.csv",
    "ISPY1": "training_metrics_log_ispy1.csv",
    "ISPY2": "training_metrics_log_ispy2.csv",
    "NACT":  "training_metrics_log_nact.csv",
}

# Paleta Okabe-Ito (colorblind friendly)
CORES = {
    "DUKE":  "#0072B2",
    "ISPY1": "#D55E00",
    "ISPY2": "#009E73",
    "NACT":  "#CC79A7",
}

COL_EPOCH = "Epoch"

COL_TRAIN_LOSS = "Train_Loss"
COL_VAL_LOSS   = "Val_Loss"

COL_TRAIN_DICE = "Train_Dice"
COL_VAL_DICE   = "Val_Dice"

COL_CPU_TIME = "CPU_Time_sec"
COL_PEAK_RAM = "Peak_RAM_MB"
COL_DURATION = "Duration_sec"

DPI = 300
FIGSIZE = (14, 6)


def carregar_dados():
    """Carrega todos os CSVs existentes."""
    dados = {}

    for nome, caminho in ARQUIVOS.items():
        if os.path.exists(caminho):
            dados[nome] = pd.read_csv(caminho)
            print(f"[OK] {nome}: {len(dados[nome])} épocas")
        else:
            print(f"[FALTOU] {nome}: {caminho}")

    return dados


def plotar_metricas(dados):
    fig, (ax_loss, ax_dice) = plt.subplots(
        1,
        2,
        figsize=FIGSIZE,
        sharex=False
    )

    # ============================================================
    # LOSS
    # ============================================================
    for nome, df in dados.items():

        cor = CORES[nome]
        epocas = df[COL_EPOCH]

        # Treino (pontilhado)
        ax_loss.plot(
            epocas,
            df[COL_TRAIN_LOSS],
            color=cor,
            linestyle=":",
            linewidth=2,
        )

        # Validação (contínuo)
        ax_loss.plot(
            epocas,
            df[COL_VAL_LOSS],
            color=cor,
            linestyle="-",
            linewidth=2,
        )

    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, linestyle="--", alpha=0.4)

    # ============================================================
    # DICE
    # ============================================================
    for nome, df in dados.items():

        cor = CORES[nome]
        epocas = df[COL_EPOCH]

        # Treino (pontilhado)
        ax_dice.plot(
            epocas,
            df[COL_TRAIN_DICE],
            color=cor,
            linestyle=":",
            linewidth=2,
        )

        # Validação (contínuo)
        ax_dice.plot(
            epocas,
            df[COL_VAL_DICE],
            color=cor,
            linestyle="-",
            linewidth=2,
        )


    ax_dice.set_xlabel("Epoch")
    ax_dice.set_ylabel("Dice Score")
    ax_dice.grid(True, linestyle="--", alpha=0.4)

    # ============================================================
    # LEGENDA DOS DATASETS
    # ============================================================

    handles_datasets = [
        Line2D(
            [0], [0],
            color=CORES[nome],
            lw=2,
            label=nome
        )
        for nome in dados.keys()
    ]

    ax_loss.legend(
        handles=handles_datasets,
        loc="upper left",
        frameon=True,
        fontsize=10
    )

    ax_dice.legend(
        handles=handles_datasets,
        loc="upper left",
        frameon=True,
        fontsize=10
    )

    # ============================================================
    # EXPLICAÇÃO DOS ESTILOS DAS LINHAS
    # ============================================================


    plt.tight_layout(rect=[0, 0.08, 1, 1])

    plt.savefig(
        "metricas_treinamento.png",
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.close()

    print("[SALVO] metricas_treinamento.png")

def plotar_recursos(dados):

    fig, (ax_cpu, ax_ram) = plt.subplots(
        1,
        2,
        figsize=FIGSIZE
    )

    # ============================================================
    # CPU TIME
    # ============================================================

    for nome, df in dados.items():

        ax_cpu.plot(
            df[COL_EPOCH],
            df[COL_CPU_TIME],
            color=CORES[nome],
            linewidth=2
        )

    ax_cpu.set_xlabel("Epoch")
    ax_cpu.set_ylabel("CPU Time (s)")
    ax_cpu.grid(True, linestyle="--", alpha=0.4)

    # ============================================================
    # PEAK RAM
    # ============================================================

    for nome, df in dados.items():

        ax_ram.plot(
            df[COL_EPOCH],
            df[COL_PEAK_RAM],
            color=CORES[nome],
            linewidth=2
        )

    ax_ram.set_xlabel("Epoch")
    ax_ram.set_ylabel("Peak RAM (MB)")
    ax_ram.grid(True, linestyle="--", alpha=0.4)

    # ============================================================

    handles = [
        Line2D(
            [0], [0],
            color=CORES[nome],
            lw=2,
            label=nome
        )
        for nome in dados.keys()
    ]

    ax_cpu.legend(handles=handles,
                  loc="upper left",
                  fontsize=10)

    ax_ram.legend(handles=handles,
                  loc="upper left",
                  fontsize=10)

    plt.tight_layout()

    plt.savefig(
        "metricas_recursos.png",
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.close()

    print("[SALVO] metricas_recursos.png")

def salvar_tempo_medio(dados):

    resultados = []

    for nome, df in dados.items():

        resultados.append({
            "Dataset": nome,
            "Average_Duration_sec": df[COL_DURATION].mean(),
            "Average_CPU_Time": df[COL_CPU_TIME].mean(),
            "Average_RAM" : df[COL_PEAK_RAM].mean()
        })

    df_saida = pd.DataFrame(resultados)

    df_saida.to_csv(
        "average_metrics_per_dataset.csv",
        index=False,
        float_format="%.3f"
    )

    print("[SALVO] average_metrics_per_dataset.csv")

def main():

    dados = carregar_dados()
    

    if not dados:
        print("Nenhum CSV encontrado.")
        return

    plotar_metricas(dados)
    salvar_tempo_medio(dados)


if __name__ == "__main__":
    main()