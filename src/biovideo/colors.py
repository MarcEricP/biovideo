from matplotlib.colors import ListedColormap
import numpy as np




def rand_cmap(nlabels: int, *,
    type: str = "bright",
    first_color_black: bool = True,
    last_color_black: bool = False,
    second_color_red: bool = False,
    seed: int | None = None) -> ListedColormap:
    """Generate a categorical random colormap similar to common gists.
    Parameters
    ---------
    nlabels : number of categories (0 is typically background)
    type : "bright" (high saturation) or "soft" (pastel)
    first_color_black : make index 0 black
    last_color_black : make last index black
    second_color_red : make index 1 red
    seed : RNG seed
    """
    rng = np.random.default_rng(seed)
    if nlabels <= 0:
        return ListedColormap([[0, 0, 0,0]])


    if type == "bright":
        H = rng.random(nlabels)
        S = rng.uniform(0.7, 1.0, nlabels)
        V = rng.uniform(0.8, 1.0, nlabels)
        import colorsys
        cols = [colorsys.hsv_to_rgb(h, s, v) for h, s, v in zip(H, S, V)]
        cols = [a + (1,) for a in cols]
    elif type == "soft":
        cols = rng.uniform(0.6, 0.95, size=(nlabels, 3)).tolist()
        cols = [a + (1,) for a in cols]
    else:
        raise ValueError("type must be 'bright' or 'soft'")


    if first_color_black and nlabels > 0:
        cols[0] = (0.0, 0.0, 0.0, 0)
    if second_color_red and nlabels > 1:
        cols[1] = (1.0, 0.1, 0.1, 1)
    if last_color_black and nlabels > 2:
        cols[-1] = (0.0, 0.0, 0.0, 1)


    return ListedColormap(cols)