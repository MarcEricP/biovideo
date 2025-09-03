from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

from .ffmpeg_utils import get_ffmpeg_path
from .colors import rand_cmap


def _prepare_ffmpeg():
    ffmpeg_path = get_ffmpeg_path()
    plt.rcParams["animation.ffmpeg_path"] = ffmpeg_path


def _pil_font(fontsize: int, font_path: str | None = None):
    if font_path:
        try:
            return ImageFont.truetype(font_path, fontsize)
        except Exception:
            pass
    try:
        prop = fm.FontProperties(size=fontsize)
        fp = fm.findfont(prop)
        return ImageFont.truetype(fp, fontsize)
    except Exception:
        return ImageFont.load_default()


def make_movie(
    movie: np.ndarray,
    save_path: str,
    mask: np.ndarray | None = None,
    label_mask: bool = False,
    num_unit_scale: float = 1,
    scale: float = 16,
    space_unit: str = "\u03bcm",
    scale_width: float = 10,
    time_interval: float = 5,
    time_unit: str = "sec",
    fps: int = 10,
    cmap: str = "gray",
    h_figsize: float = 5,
    fontsize: int = 15,
    first_label_red: bool = False,
    label_font_path: str | None = None,
) -> None:
    """Create an MP4 movie from a 3D array (T,H,W). Optionally overlay a label mask.

    Parameters
    ----------
    mask : integer labels (0 = background). If `label_mask=True`, draw indices.
    num_unit_scale : physical unit count for the scale bar label (e.g., micrometers)
    scale : number of pixels corresponding to `num_unit_scale` units
    """
    _prepare_ffmpeg()

    T, H, W = movie.shape
    aspect_ratio = H / W
    plt.rcParams["figure.figsize"] = (h_figsize / aspect_ratio, h_figsize)

    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    ax.axis("off")

    img = ax.imshow(movie[0], cmap=cmap)

    fontprops = fm.FontProperties(size=fontsize)
    scalebar = AnchoredSizeBar(
        ax.transData,
        scale * num_unit_scale,
        f"{num_unit_scale} {space_unit}",
        "lower right",
        pad=0.1,
        color="white",
        frameon=False,
        size_vertical=scale_width,
        fontproperties=fontprops,
    )
    ax.add_artist(scalebar)

    rgba_mask = None
    if mask is not None:
        if not label_mask:
            new_cmap = rand_cmap(mask.max() + 1, type='bright', first_color_black=True, last_color_black=False, second_color_red=first_label_red)
            img_mask = ax.imshow(mask[0], cmap=new_cmap)
        else:
            new_cmap = rand_cmap(mask.max() + 1, type='bright', first_color_black=True, last_color_black=False, second_color_red=first_label_red)
            rgba_mask = np.zeros(mask.shape + (4,), dtype=float)
            pil_font = _pil_font(fontsize, label_font_path)
            for t in range(mask.shape[0]):
                rgba = new_cmap(mask[t])
                final_img = (255 * rgba).astype(np.uint8)
                pil_final = PILImage.fromarray(final_img)
                draw = ImageDraw.Draw(pil_final)
                values = np.unique(mask[t])
                values = values[values != 0]
                for val in values:
                    rr, cc = np.nonzero(mask[t] == val)
                    if rr.size == 0:
                        continue
                    centroid = (int(np.round(cc.mean()) - 30), int(np.round(rr.mean()) - 30))
                    color = tuple(int(255 * c) for c in new_cmap(val)[:3])
                    draw.text(centroid, f"{val - 1}", fill=color, font=pil_font)
                rgba_mask[t] = np.array(pil_final) / 255.0
            img_mask = ax.imshow(rgba_mask[0])

    time_text = ax.text(int(0.1 * H), int(0.1 * W * aspect_ratio), '', fontsize=fontsize, color='white')

    def update_frame(frame: int):
        img.set_data(movie[frame])
        if mask is not None:
            img_mask.set_data(rgba_mask[frame] if label_mask else mask[frame])
        tval = time_interval * frame
        if float(tval).is_integer():
            time_text.set_text(f"{int(tval)} {time_unit}")
        else:
            time_text.set_text(f"{tval:0.2f} {time_unit}")

    ani = animation.FuncAnimation(fig=fig, func=update_frame, frames=T, interval=int(1000 / fps))
    ani.save(filename=save_path, writer='ffmpeg')
    plt.close(fig)


def make_movie_and_plot(
    movie: np.ndarray,
    save_path: str,
    time: np.ndarray,
    metric: np.ndarray,
    xlabel: str = "",
    ylabel: str = "",
    mask: np.ndarray | None = None,
    num_unit_scale: float = 1,
    scale: float = 16,
    space_unit: str = "\u03bcm",
    constant_amplitude: float | None = None,
    y_not_under_zero: bool = True,
    scale_width: float = 10,
    time_interval: float = 5,
    time_unit: str = "sec",
    fps: int = 10,
    cmap: str = "gray",
    h_figsize: float = 20,
    fontsize: int = 15,
    first_label_red: bool = False,
) -> None:
    _prepare_ffmpeg()

    T, H, W = movie.shape
    aspect_ratio = H / W

    plt.rcParams["figure.figsize"] = (h_figsize, h_figsize * 3 / 8)
    plt.rcParams.update({'font.size': 20})

    fig, ax = plt.subplots(1, 2)
    fig.subplots_adjust(left=0.01, bottom=0.1, right=0.99, top=0.99)

    ax[0].axis("off")
    img = ax[0].imshow(movie[0], cmap=cmap)

    fontprops = fm.FontProperties(size=fontsize)
    scalebar = AnchoredSizeBar(
        ax[0].transData,
        scale * num_unit_scale,
        f"{num_unit_scale} {space_unit}",
        "lower right",
        pad=0.1,
        color='white',
        frameon=False,
        size_vertical=scale_width,
        fontproperties=fontprops,
    )
    ax[0].add_artist(scalebar)

    if mask is not None:
        new_cmap = rand_cmap(mask.max() + 1, type='bright', first_color_black=True, second_color_red=first_label_red, last_color_black=False)
        img_mask = ax[0].imshow(mask[0], cmap=new_cmap)

    time_text = ax[0].text(int(0.1 * H), int(0.1 * W * aspect_ratio), '', fontsize=fontsize, color='white')

    ax[1].plot(time, metric, zorder=0)
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel(ylabel)
    scat = ax[1].scatter(time[0], metric[0], zorder=2, s=200)

    if constant_amplitude is not None:
        amp = float(metric.max() - metric.min())
        ymin = float(metric.min() - (constant_amplitude - amp) / 2)
        ymax = float(metric.max() + (constant_amplitude - amp) / 2)
        if ymin < 0 and y_not_under_zero:
            diff = 0 - ymin
            ymin += diff
            ymax += diff
        ax[1].set_ylim([ymin, ymax])

    def update_frame(frame: int):
        img.set_data(movie[frame])
        if mask is not None:
            img_mask.set_data(mask[frame])
        tval = time_interval * frame
        if float(tval).is_integer():
            time_text.set_text(f"{int(tval)} {time_unit}")
        else:
            time_text.set_text(f"{tval:0.2f} {time_unit}")
        try:
            scat.set_offsets(np.array([[time[frame], metric[frame]]]))
        except Exception:
            pass

    ani = animation.FuncAnimation(fig=fig, func=update_frame, frames=T, interval=int(1000 / fps))
    ani.save(filename=save_path, writer='ffmpeg')
    plt.close(fig)