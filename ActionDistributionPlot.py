from collections import Counter

import matplotlib.pyplot as plt

plt.style.use("bmh")


def actionDistributionPlot():
    actionDistribution = {}
    xLabels = ["Palavra Inicial", "Máxima Correspondência de Palavras", "Máxima Correspondência de Letras", "Máximo Compartilhamento de Letras", "Menor Distância de Levenshtein", "Seleção Aleatória"]
    allActions = Counter(actionDistribution[(0, 0)])
    plt.xticks(range(len(xLabels)), xLabels, rotation=90)
    plt.bar(allActions.keys(), allActions.values())
    plt.ylabel("Chamadas")
    plt.yscale("linear")
    plt.tight_layout()
    plt.savefig("log/ActionDistribution.png")
    plt.show()


if __name__ == "__main__":
    actionDistributionPlot()
