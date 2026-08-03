import re
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker

# ============================================================
# CONFIGURAÇÃO
# ============================================================

ARQUIVOS = {
    "Volumétrico": "fed.txt",
    "Unificado": "fed(1).txt",
}

CORES = {
    "Volumétrico": "#D55E00",
    "Unificado": "#0072B2",
}

DPI = 300
FIGSIZE = (14, 6)


# ============================================================
# LEITURA DOS LOGS
# ============================================================

def carregar_log(caminho):

    with open(caminho, "r", encoding="utf-8") as f:
        texto = f.read()

    rounds = []
    train_loss = []
    val_loss = []
    val_dice = []

    padrao = re.compile(
        r"\[ROUND\s+(\d+)/\d+\].*?"
        r"train_loss':\s*([0-9eE.+-]+).*?"
        r"loss_agregado':\s*([0-9eE.+-]+),\s*"
        r"'dice_agregado':\s*([0-9eE.+-]+)",
        re.S
    )

    for r, tl, vl, vd in padrao.findall(texto):

        rounds.append(int(r))
        train_loss.append(float(tl))
        val_loss.append(float(vl))
        val_dice.append(float(vd))

    return {
        "Round": rounds,
        "Train_Loss": train_loss,
        "Val_Loss": val_loss,
        "Val_Dice": val_dice,
    }


def carregar_dados():

    dados = {}

    for nome, arquivo in ARQUIVOS.items():

        dados[nome] = carregar_log(arquivo)

        print(f"[OK] {nome}: {len(dados[nome]['Round'])} rounds")

    return dados




def plotar_metricas(dados):

    fig, (ax_loss, ax_dice) = plt.subplots(
        1,
        2,
        figsize=FIGSIZE
    )

    # ==========================================================
    # LOSS
    # ==========================================================

    for nome, d in dados.items():

        ax_loss.plot(
            d["Round"],
            d["Val_Loss"],
            color=CORES[nome],
            linewidth=2.5,
            label=nome
        )

    ax_loss.set_xlabel("Round")
    ax_loss.set_ylabel("Validation Loss")

    ax_loss.grid(True, linestyle="--", alpha=0.35)

    # Apenas números inteiros no eixo X
    ax_loss.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # pequena margem para não encostar nas bordas
    ax_loss.margins(x=0.02, y=0.08)

    # ==========================================================
    # DICE
    # ==========================================================

    for nome, d in dados.items():

        ax_dice.plot(
            d["Round"],
            d["Val_Dice"],
            color=CORES[nome],
            linewidth=2.5,
            label=nome
        )

    ax_dice.set_xlabel("Round")
    ax_dice.set_ylabel("Validation Dice")

    ax_dice.grid(True, linestyle="--", alpha=0.35)

    ax_dice.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax_dice.margins(x=0.02, y=0.08)

    # ==========================================================
    # LEGENDA
    # ==========================================================

    handles = [
        Line2D([0], [0], color=CORES["Volumétrico"], lw=2.5, label="Volumétrico"),
        Line2D([0], [0], color=CORES["Unificado"], lw=2.5, label="Unificado"),
    ]

    ax_loss.legend(handles=handles, loc="upper right", frameon=True)
    ax_dice.legend(handles=handles, loc="lower right", frameon=True)

    plt.tight_layout()

    plt.savefig(
        "metricas_federado.png",
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.close()

    print("[SALVO] metricas_federado.png")
    
def main():

    dados = carregar_dados()

    plotar_metricas(dados)


if __name__ == "__main__":
    main()