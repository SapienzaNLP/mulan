#
# Copyright 2020 SapienzaNLP research group (http://nlp.uniroma1.it/)
# Authors: Edoardo Barba, Luigi Procopio, Niccolò Campolungo, Tommaso Pasini, Roberto Navigli
# GitHub Repo: https://github.com/SapienzaNLP/mulan
# License: Attribution-NonCommercial-ShareAlike 4.0 International
#

import matplotlib.pyplot as plt
import numpy as np


def plot_heat_matrix(matrix, x_labels, y_labels, title='Heat matrix'):

    fig, ax = plt.subplots()
    im = ax.imshow(matrix)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(y_labels)))
    ax.set_yticks(np.arange(len(x_labels)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(y_labels)
    ax.set_yticklabels(x_labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.

    for i in range(len(x_labels)):
        for j in range(len(y_labels)):
            ax.text(j, i, str(matrix[i, j])[:4], ha="center", va="center", color="w")

    ax.set_title(title)

    return ax
