import os
import numpy as np
from biovideo import make_movie, make_movie_and_plot
from matplotlib import cm
from matplotlib.colors import ListedColormap

outdir = os.path.join(os.getcwd(), "_out")
os.makedirs(outdir, exist_ok=True)

movie = np.random.random((100, 128, 256))
mask = np.zeros_like(movie, dtype=int)
mask[:, 40:43, 60:120] = 5

make_movie(movie, os.path.join(outdir, "test.mp4"), mask=mask, label_mask=True, num_unit_scale=30)

# with plot
T = movie.shape[0]
time = np.arange(T)
metric = np.sin(time / 10) + np.random.normal(0, 0.05, size=T)
make_movie_and_plot(movie, os.path.join(outdir, "test_with_plot.mp4"), time, metric, "t", "metric", mask=mask, num_unit_scale=30)

# Custom categorical (index 0 transparent for background)
mask_cmap = ListedColormap([
    [0, 0, 0, 0],   # background transparent
    [1, 0, 0, 1],   # label 1 red
    [0, 1, 0, 1],   # label 2 green
    [0, 0, 1, 1],   # label 3 blue
])
make_movie(movie, os.path.join(outdir, "custom_cmap.mp4"), mask=mask, img_cmap=cm.gray, mask_cmap=mask_cmap)