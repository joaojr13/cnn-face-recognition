"""Figuras da avaliação, usadas tanto pelo relatório PNG quanto pela interface."""
import numpy as np

MEMBER_COLOR = "#2a78d6"   # azul: integrantes
EXTERNAL_COLOR = "#eb6834"  # laranja: pessoas externas


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")  # gera imagens sem abrir janelas (terminal e servidor)
    import matplotlib.pyplot as plt
    return plt


def display(name):
    return name[:1].upper() + name[1:]


def confusion_figure(labels, matrix, title):
    """Linhas: identidade correta. Colunas: resposta do modelo (incluindo Desconhecido)."""
    plt = _pyplot()
    values = np.asarray(matrix)
    # A cor usa a fração de cada linha: 1 de 1 foto aparece tão forte quanto 48 de 50.
    totals = values.sum(axis=1, keepdims=True)
    shares = np.divide(values, totals, out=np.zeros(values.shape, dtype=float), where=totals > 0)
    fig, ax = plt.subplots(figsize=(7, 5.4))
    ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
    names = [display(x) for x in labels]
    ax.set_xticks(range(len(labels)), names, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), [f"{name} ({int(total)})" for name, total in zip(names, totals[:, 0])])
    for row in range(len(labels)):
        for col in range(len(labels)):
            ax.text(col, row, str(values[row, col]), ha="center", va="center",
                    color="white" if shares[row, col] > .55 else "#1a1a19")
    ax.set(xlabel="Resposta do modelo", ylabel="Identidade correta (fotos)", title=title)
    fig.text(0.015, 0.015, "Números: contagem de fotos. Cor: fração da linha.",
             fontsize=8, color="#6b6a64")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def score_figure(panels, threshold, provisional=None):
    """Maior pontuação de cada foto, separando integrantes e externos.

    panels: lista de (título, pontuações_integrantes, pontuações_externos).
    A linha contínua é o limiar calibrado; a tracejada, o limiar provisório anterior.
    """
    plt = _pyplot()
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 3.4), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (title, members, externals) in zip(axes, panels):
        for row, (values, color, marker, name) in enumerate([
                (externals, EXTERNAL_COLOR, "^", "Externos"),
                (members, MEMBER_COLOR, "o", "Integrantes")]):
            values = np.asarray(values, dtype=float)
            jitter = rng.uniform(-0.18, 0.18, len(values))
            ax.scatter(values, row + jitter, s=46, color=color, marker=marker,
                       edgecolors="white", linewidths=1.2, zorder=3, label=f"{name} ({len(values)})")
        ax.axvline(threshold, color="#1a1a19", linewidth=2, zorder=2)
        ax.text(threshold, 1.42, f" limiar {threshold:.3f}", fontsize=8, va="center")
        if provisional is not None and abs(provisional - threshold) > 1e-9:
            ax.axvline(provisional, color="#6b6a64", linewidth=1.5, linestyle="--", zorder=2)
            side = "right" if provisional < threshold else "left"
            ax.text(provisional, -0.45, f" provisório {provisional:.2f} ", fontsize=8,
                    va="center", ha=side, color="#6b6a64")
        ax.set_yticks([0, 1], ["Externos", "Integrantes"])
        ax.set_ylim(-0.65, 1.6)
        ax.set_xlim(0, 1.02)
        ax.set_xlabel("Maior pontuação softmax")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", color="#e4e3dd", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def threshold_curve_figure(rows, threshold, provisional=None, target_far=None):
    """Efeito do limiar na validação: integrantes rejeitados × externos aceitos.

    rows: saídas de avaliacao_facial.threshold_effect em ordem crescente de limiar.
    """
    plt = _pyplot()
    x = [row["threshold"] for row in rows]
    rejected = [100 * row["rejection_rate"] for row in rows]
    accepted = [100 * row["false_accept_rate"] for row in rows]
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    ax.step(x, accepted, where="post", color=EXTERNAL_COLOR, linewidth=2, label="Externos aceitos (%)")
    ax.step(x, rejected, where="post", color=MEMBER_COLOR, linewidth=2, label="Integrantes rejeitados (%)")
    if target_far is not None:
        ax.axhline(100 * target_far, color=EXTERNAL_COLOR, linewidth=1, linestyle=":", alpha=.8)
        ax.text(x[0], 100 * target_far + 1.5, f"máximo aceito de externos: {target_far:.0%}",
                fontsize=8, color="#6b6a64")
    ax.axvline(threshold, color="#1a1a19", linewidth=2)
    ax.text(threshold, 96, f" calibrado {threshold:.3f}", fontsize=8, va="top")
    if provisional is not None and abs(provisional - threshold) > 1e-9:
        ax.axvline(provisional, color="#6b6a64", linewidth=1.5, linestyle="--")
        side = "right" if provisional < threshold else "left"
        ax.text(provisional, 96, f" provisório {provisional:.2f} ", fontsize=8, va="top",
                ha=side, color="#6b6a64")
    ax.set(xlim=(x[0], x[-1]), ylim=(0, 100), xlabel="Limiar", ylabel="Fotos da validação (%)",
           title="Efeito do limiar na validação")
    ax.grid(axis="y", color="#e4e3dd", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="center left", fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def save(fig, path):
    plt = _pyplot()
    fig.savefig(path, dpi=160)
    plt.close(fig)
