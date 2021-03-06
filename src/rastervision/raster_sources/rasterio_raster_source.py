import tempfile

import numpy as np

from rastervision.core.raster_source import RasterSource
from rastervision.core.box import Box
from rastervision.utils.files import download_if_needed


def load_window(image_dataset, window=None):
    """Load a window of an image from a TIFF file.

    Args:
        window: ((row_start, row_stop), (col_start, col_stop)) or
        ((y_min, y_max), (x_min, x_max))
    """
    im = image_dataset.read(window=window, boundless=True)
    im = np.transpose(im, axes=[1, 2, 0])
    return im


class RasterioRasterSource(RasterSource):
    def __init__(self, raster_transformer):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.image_dataset = self.build_image_dataset()
        super().__init__(raster_transformer)

    def build_image_dataset(self):
        pass

    def get_extent(self):
        return Box(
            0, 0, self.image_dataset.height, self.image_dataset.width)

    def _get_chip(self, window):
        return load_window(self.image_dataset, window.rasterio_format())
