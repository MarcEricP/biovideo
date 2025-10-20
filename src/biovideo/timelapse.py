import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.font_manager import FontProperties as _FP
import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont
from typing import Optional, Union
from .colors import rand_cmap
from .anim import _resolve_cmap,_pil_font
ColormapLike = Union[str, mcolors.Colormap]

# Assumes you already have these helpers in your codebase (as in make_movie)
# _resolve_cmap(name_or_obj, fallback)
# _pil_font(size, path)

def make_time_lapse(
        movie: np.ndarray,
        list_frames,
        n_rows: int,
        mask: np.ndarray | None = None,
        label_mask: bool = False,
        num_unit_scale: float = 1,
        scale: float = 16,
        space_unit: str = "μm",
        scale_width: float = 10,
        time_interval: float = 5,
        time_unit: str = "sec",
        time_offset = 0,
        is_hour = False,
        pos_time = (0.1, 0.1),
        img_cmap: ColormapLike = "gray",
        mask_cmap: Optional[ColormapLike] = None,
        h_figsize: float = 5,
        fontsize: int = 15,
        first_label_red: bool = False,
        label_font_path: str | None = None,
):
    """
    Arrange selected frames into a grid (n_rows x n_cols). Each tile shows a timestamp;
    the last tile also shows a scale bar. Optional mask overlay (label or continuous).
    Returns the matplotlib.figure.Figure.
    """
    frames = list(list_frames)
    if len(frames) == 0:
        raise ValueError("list_frames is empty.")
    T, H, W = movie.shape[:3]
    if any(f < 0 or f >= T for f in frames):
        raise IndexError("A frame in list_frames is out of bounds.")

    n_cols = math.ceil(len(frames) / n_rows)
    panel_aspect = W / H
    fig_h = float(h_figsize)
    fig_w = fig_h * (n_cols / n_rows) * panel_aspect
    
    
    fig, axs = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w, fig_h),
        squeeze=False,
        gridspec_kw={'wspace': 0, 'hspace': 0},
        constrained_layout=False
    )

    for ax in axs.flat:
        ax.set_axis_off()
        ax.margins(0)
        ax.set_aspect('equal')

    img_cmap_resolved = _resolve_cmap(img_cmap, "gray")

    rgba_mask_sel = None
    if mask is not None:
        # Resolve mask cmap
        if mask_cmap is None:
            # bright random map; background 0 black; label 1 can be forced red
            default_mask_cmap = rand_cmap(int(mask.max()) + 1, type='bright',
                                          first_color_black=True, last_color_black=False,
                                          second_color_red=first_label_red)
            mask_cmap_resolved = _resolve_cmap(mask_cmap, default_mask_cmap)
        else:
            mask_cmap_resolved = _resolve_cmap(mask_cmap, None)

        # Normalize mask to (T, H, W)
        if mask.ndim == 2:
            maskT = np.repeat(mask[None, ...], T, axis=0)
        elif mask.ndim == 3:
            maskT = mask
        else:
            raise ValueError("mask must be (H,W) or (T,H,W).")

        # Build RGBA overlays only for requested frames
        rgba_mask_sel = {}
        pil_font = _pil_font(fontsize, label_font_path)
        for f in frames:
            lab = maskT[f]
            rgba = mask_cmap_resolved(lab)  # (H, W, 4), floats 0..1
            out = (255 * rgba).astype(np.uint8)
            pil_img = PILImage.fromarray(out)
            if label_mask:
                draw = ImageDraw.Draw(pil_img)
                vals = np.unique(lab)
                vals = vals[vals != 0]
                for val in vals:
                    rr, cc = np.nonzero(lab == val)
                    if rr.size == 0:
                        continue
                    cy, cx = int(np.round(rr.mean())), int(np.round(cc.mean()))
                    color = tuple(int(255 * c) for c in mask_cmap_resolved(val)[:3])
                    draw.text((cx - 30, cy - 30), f"{val - 1}", fill=color, font=pil_font)
            rgba_mask_sel[f] = np.array(pil_img, dtype=np.uint8).astype(float) / 255.0

    # ---- draw tiles ----
    fp = _FP(size=fontsize)
    aspect_ratio = H / W  # for timestamp placement compatibility with make_movie

    for idx, f in enumerate(frames):
        r, c = divmod(idx, n_cols)
        ax = axs[r, c]
        ax.set_axis_off()

        # Show image (handles grayscale or RGB)
        if movie.ndim == 3:
            ax.imshow(movie[f], cmap=img_cmap_resolved, interpolation="nearest")
        else:
            ax.imshow(movie[f], interpolation="nearest")

        # Mask overlay
        if rgba_mask_sel is not None:
            ax.imshow(rgba_mask_sel[f], interpolation="nearest", alpha=0.7)

        # Timestamp (match make_movie style: pos in pixels of the Axes extent)
        tval = time_interval * (f) + time_offset
        formatted_time = int(tval)
        if is_hour:
            n_hour = int(tval)
            remaining = int((tval - int(tval))*60)
            n_minutes = remaining if remaining > 0 else ""
            formatted_time = f"{n_hour}h{n_minutes}"
        ttxt = f"{formatted_time} {time_unit}" if float(tval).is_integer() else f"{tval:0.2f} {time_unit}"
        ax.text(int(pos_time[0] * H), int(pos_time[1] * W * aspect_ratio), ttxt,
                fontsize=fontsize, color='white',
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0., edgecolor="none"))

        # Scale bar only on the last panel
        if idx == len(frames) - 1:
            scalebar = AnchoredSizeBar(
                ax.transData,
                scale * num_unit_scale,                     
                f"{num_unit_scale} {space_unit}",
                "lower right",
                pad=0.1,
                color="white",
                frameon=False,
                size_vertical=scale_width,
                fontproperties=fp,
            )
            ax.add_artist(scalebar)

    # Hide unused axes
    total = n_rows * n_cols
    for idx in range(len(frames), total):
        r, c = divmod(idx, n_cols)
        axs[r, c].set_axis_off()

    return(fig,axs)
