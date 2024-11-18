# Functions to plot strokes and attention plots
import numpy as np
import matplotlib.pyplot as plt

def plot_stroke(stroke, save_name=None):
    fig, ax = plt.subplots()

    x = np.cumsum(stroke[:, 1])
    y = np.cumsum(stroke[:, 2])

    size_x = x.max() - x.min() + 1.
    size_y = y.max() - y.min() + 1.
    fig.set_size_inches(5. * size_x / size_y, 5.)

    cuts = np.where(stroke[:, 0] == 1)[0]
    start = 0

    for cut in cuts:
        ax.plot(x[start:cut], y[start:cut], 'k-', linewidth=3)
        start = cut + 1

    ax.axis('equal')
    ax.axis('off')

    if save_name is None:
        plt.show()
    else:
        try:
            plt.savefig(save_name, bbox_inches='tight', pad_inches=0.5)
        except Exception as e:
            print(f"Error saving image: {save_name}. Error: {str(e)}")

    plt.close()

def attention_plot(phis):
    phis = phis / np.sum(phis, axis=0, keepdims=True)

    plt.figure(figsize=(12, 6))
    plt.xlabel('Handwriting Generation')
    plt.ylabel('Text Scanning')

    plt.imshow(phis, cmap='hot', interpolation='nearest', aspect='auto')
    plt.colorbar(label='Attention Weight')
    plt.show()

    