# biovideo
Little package for combining timelapse and mask as numpy array into mp4 videos. Also provides a scalebar and timestamp. Auto-detects FFmpeg and downloads a static build if missing.

## Install
```bash
pip install -e .
```

## Usage
```python
from biovideo import make_movie
# see examples/demo_generate.py
```

## Acknowledgments
This project builds upon the work of:

- [NumPy](https://numpy.org/) (BSD-3)
- [Matplotlib](https://matplotlib.org/) (Matplotlib License)
- [Pillow](https://python-pillow.org/) (PIL/Pillow License)
- [appdirs](https://github.com/ActiveState/appdirs) (MIT)
- [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg) (BSD-2)
- [FFmpeg](https://ffmpeg.org/) (LGPL v2.1+ or GPL, depending on build)

FFmpeg binaries are downloaded automatically if not found. Please see the [FFmpeg license terms](https://ffmpeg.org/legal.html).

<!--## Cite this work
```bibtex
@software{Perrin_shrugnet_2025,
author = {Perrin, Marc-Eric},
license = {MIT},
month = sep,
title = {{shrugnet}},
url = {https://github.com/MarcEricP/shrugnet},
version = {0.1.0},
year = {2025}
}
```-->