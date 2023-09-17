import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch


def set_ticks(tick_values, tick_labels, tick_colors):
    plt.xticks(ticks=tick_values, labels=tick_labels)
    ticklabels = plt.gca().get_xticklabels()
    for ticklabel, tickcolor in zip(ticklabels, tick_colors):
        ticklabel.set_color(tickcolor)


def get_next_color(colormap, n_colors=None):
    if n_colors is None:
        n_colors = len(colormap.colors) if hasattr(colormap, "colors") else 256
    for idx in np.linspace(0, 1, n_colors, endpoint=False):
        yield colormap(idx)


def ema_smooth(values, alpha=0.1):
    ema_values = []
    ema = values[0]
    for value in values:
        value_tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value)
        if torch.isnan(value_tensor):
            print("NAN")
            continue
        ema = (1 - alpha) * ema + alpha * value
        ema_values.append(ema)
    return ema_values


def rescale_colormap(cmap_name, new_min=0.25, new_mid=0.5, new_max=1.0):
    cmap = plt.get_cmap(cmap_name)
    start_color = cmap(new_min)
    mid_color = cmap(new_mid)
    end_color = cmap(new_max)
    colors = [start_color, mid_color, end_color]
    cmap = LinearSegmentedColormap.from_list(cmap_name + "_modified", colors)
    return cmap
