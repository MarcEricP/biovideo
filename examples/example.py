import os
import numpy as np
from biovideo import make_movie
from matplotlib.colors import ListedColormap
from skimage import io,morphology

outdir = os.path.join(os.getcwd(), "examples/generated_movies")
os.makedirs(outdir, exist_ok=True)

movie = io.imread("examples/data/movie.tif")
mask = io.imread("examples/data/skel_stack.tif")
for t in range(mask.shape[0]):
    mask[t] = morphology.dilation(mask[t])
# Custom colormap (index 0 transparent for background)
mask_cmap = ListedColormap([
    [0, 0, 0, 0],   # background transparent
    [1, 0, 0, 1],   # label 1 red
    [0, 0, 1, 1],   # label 2 blue
])

make_movie(
    movie, 
    os.path.join(outdir, "movie.mp4"),
    mask=mask, 
    label_mask=False, 
    num_unit_scale=3,
    scale = 16,
    space_unit="µm",
    time_interval=1,
    time_unit="min",
    pos_time = (0.01,0.05),
    fps=5,
    img_cmap = "viridis",
    mask_cmap = mask_cmap,
    fontsize=20,
    scale_width=3,
    )

# with plot
time = np.arange(movie.shape[0])
metric = np.sin(time / 10) + np.random.normal(0, 0.05, size=movie.shape[0])

make_movie(
    movie, 
    os.path.join(outdir, os.path.join(outdir, "movie_with_plot.mp4")),
    time=time,
    metric=metric, 
    xlabel="t", 
    ylabel="metric",
    mask=mask,  
    num_unit_scale=3,
    scale = 16,
    space_unit="µm",
    time_interval=1,
    time_unit="min",
    pos_time = (0.05,0.1),
    fps=5,
    img_cmap = "gray",
    mask_cmap = mask_cmap,
    fontsize=45,
    scale_width=3,
    h_figsize=20,
    )
