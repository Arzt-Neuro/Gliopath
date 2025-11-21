#%%
import logging
import shutil
import tempfile
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable
import skimage.filters

import pandas as pd
from openslide import OpenSlide
from matplotlib import collections, patches, pyplot as plt
from monai.data.wsi_reader import WSIReader
from tqdm import tqdm
import PIL
import os

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
import numpy as np
from scipy import ndimage

from monai.transforms.transform import MapTransform
#%%
@dataclass(frozen=True)
class Box:
    """Utility class representing rectangular regions in 2D images.

    :param x: Horizontal coordinate of the top-left corner.
    :param y: Vertical coordinate of the top-left corner.
    :param w: Box width.
    :param h: Box height.
    :raises ValueError: If either `w` or `h` are <= 0.
    """
    x: int
    y: int
    w: int
    h: int

    def __post_init__(self) -> None:
        if self.w <= 0:
            raise ValueError(f"Width must be strictly positive, received {self.w}")
        if self.h <= 0:
            raise ValueError(f"Height must be strictly positive, received {self.w}")

    def __add__(self, shift: Sequence[int]) -> 'Box':
        """Translates the box's location by a given shift.

        :param shift: A length-2 sequence containing horizontal and vertical shifts.
        :return: A new box with updated `x = x + shift[0]` and `y = y + shift[1]`.
        :raises ValueError: If `shift` does not have two elements.
        """
        if len(shift) != 2:
            raise ValueError("Shift must be two-dimensional")
        return Box(x=self.x + shift[0],
                   y=self.y + shift[1],
                   w=self.w,
                   h=self.h)

    def __mul__(self, factor: float) -> 'Box':
        """Scales the box by a given factor, e.g. when changing resolution.

        :param factor: The factor by which to multiply the box's location and dimensions.
        :return: The updated box, with location and dimensions rounded to `int`.
        """
        return Box(x=int(self.x * factor),
                   y=int(self.y * factor),
                   w=int(self.w * factor),
                   h=int(self.h * factor))

    def __rmul__(self, factor: float) -> 'Box':
        """Scales the box by a given factor, e.g. when changing resolution.

        :param factor: The factor by which to multiply the box's location and dimensions.
        :return: The updated box, with location and dimensions rounded to `int`.
        """
        return self * factor

    def __truediv__(self, factor: float) -> 'Box':
        """Scales the box by a given factor, e.g. when changing resolution.

        :param factor: The factor by which to divide the box's location and dimensions.
        :return: The updated box, with location and dimensions rounded to `int`.
        """
        return self * (1. / factor)

    def add_margin(self, margin: int) -> 'Box':
        """Adds a symmetric margin on all sides of the box.

        :param margin: The amount by which to enlarge the box.
        :return: A new box enlarged by `margin` on all sides.
        """
        return Box(x=self.x - margin,
                   y=self.y - margin,
                   w=self.w + 2 * margin,
                   h=self.h + 2 * margin)

    def clip(self, other: 'Box') -> Optional['Box']:
        """Clips a box to the interior of another.

        This is useful to constrain a region to the interior of an image.

        :param other: Box representing the new constraints.
        :return: A new constrained box, or `None` if the boxes do not overlap.
        """
        x0 = max(self.x, other.x)
        y0 = max(self.y, other.y)
        x1 = min(self.x + self.w, other.x + other.w)
        y1 = min(self.y + self.h, other.y + other.h)
        try:
            return Box(x=x0, y=y0, w=x1 - x0, h=y1 - y0)
        except ValueError:  # Empty result, boxes don't overlap
            return None

    def to_slices(self) -> Tuple[slice, slice]:
        """Converts the box to slices for indexing arrays.

        For example: `my_2d_array[my_box.to_slices()]`.

        :return: A 2-tuple with vertical and horizontal slices.
        """
        return (slice(self.y, self.y + self.h),
                slice(self.x, self.x + self.w))

    @staticmethod
    def from_slices(slices: Sequence[slice]) -> 'Box':
        """Converts a pair of vertical and horizontal slices into a box.

        :param slices: A length-2 sequence containing vertical and horizontal `slice` objects.
        :return: A box with corresponding location and dimensions.
        """
        vert_slice, horz_slice = slices
        return Box(x=horz_slice.start,
                   y=vert_slice.start,
                   w=horz_slice.stop - horz_slice.start,
                   h=vert_slice.stop - vert_slice.start)
#%%
def _get_image_size(img, size=None, level=None, location=(0, 0), backend="openslide"):
    max_size = []
    downsampling_factor = []
    if backend == "openslide":
        downsampling_factor = img.level_downsamples[level]
        max_size = img.level_dimensions[level][::-1]
    elif backend == "cucim":
        downsampling_factor = img.resolutions["level_downsamples"][level]
        max_size = img.resolutions["level_dimensions"][level][::-1]
    elif backend == "tifffile":
        level0_size = img.pages[0].shape[:2]
        max_size = img.pages[level].shape[:2]
        downsampling_factor = np.mean([level0_size[i] / max_size[i] for i in range(len(max_size))])

    # subtract the top left corner of the patch from maximum size
    level_location = [round(location[i] / downsampling_factor) for i in range(len(location))]
    size = [max_size[i] - level_location[i] for i in range(len(max_size))]

    return size
#%%
def load_slide_at_level(reader: WSIReader, slide_obj: OpenSlide, level: int) -> np.ndarray:
    """Load full slide array at the given magnification level.

    This is a manual workaround for a MONAI bug (https://github.com/Project-MONAI/MONAI/issues/3415)
    fixed in a currently unreleased PR (https://github.com/Project-MONAI/MONAI/pull/3417).

    :param reader: A MONAI `WSIReader` using OpenSlide backend.
    :param slide_obj: The OpenSlide image object returned by `reader.read(<image_file>)`.
    :param level: Index of the desired magnification level as defined in the `slide_obj` headers.
    :return: The loaded image array in (C, H, W) format.
    """
    size = _get_image_size(slide_obj, level=level)
    img_data, meta_data = reader.get_data(slide_obj, size=size, level=level)
    logging.info(f"img: {img_data.dtype} {img_data.shape}, metadata: {meta_data}")

    return img_data
#%%
def get_bounding_box(mask: np.ndarray) -> Box:
    """Extracts a bounding box from a binary 2D array.

    :param mask: A 2D array with 0 (or `False`) as background and >0 (or `True`) as foreground.
    :return: The smallest box covering all non-zero elements of `mask`.
    :raises TypeError: When the input mask has more than two dimensions.
    :raises RuntimeError: When all elements in the mask are zero.
    """
    if mask.ndim != 2:
        raise TypeError(f"Expected a 2D array but got an array with shape {mask.shape}")

    slices = ndimage.find_objects((mask > 0).astype(int))
    if not slices:
        raise RuntimeError("The input mask is empty")
    assert len(slices) == 1

    return Box.from_slices(slices[0])
#%%
class LoadROId(MapTransform):
    """Transform that loads a pathology slide, cropped to the foreground bounding box (ROI).

    Operates on dictionaries, replacing the file paths in `image_key` with the
    respective loaded arrays, in (C, H, W) format. Also adds the following meta-data entries:
    - `'location'` (tuple): top-right coordinates of the bounding box
    - `'size'` (tuple): width and height of the bounding box
    - `'level'` (int): chosen magnification level
    - `'scale'` (float): corresponding scale, loaded from the file
    """

    def __init__(self, image_reader: WSIReader, image_key: str = "image", level: int = 0,
                 margin: int = 0, foreground_threshold: Optional[float] = None) -> None:
        """
        :param reader: An instance of MONAI's `WSIReader`.
        :param image_key: Image key in the input and output dictionaries.
        :param level: Magnification level to load from the raw multi-scale files.
        :param margin: Amount in pixels by which to enlarge the estimated bounding box for cropping.
        """
        super().__init__([image_key], allow_missing_keys=False)
        self.image_reader = image_reader
        self.image_key = image_key
        self.level = level
        self.margin = margin
        self.foreground_threshold = foreground_threshold

    def _get_bounding_box(self, slide_obj: OpenSlide) -> Box:
        # Estimate bounding box at the lowest resolution (i.e. highest level)
        highest_level = slide_obj.level_count - 1
        slide = load_slide_at_level(self.image_reader, slide_obj, level=highest_level)

        if slide_obj.level_count == 1:
            logging.warning(f"Only one image level found. segment_foregound will use a lot of memory.")

        foreground_mask, threshold = segment_foreground(slide, self.foreground_threshold)

        scale = slide_obj.level_downsamples[highest_level]
        bbox = scale * get_bounding_box(foreground_mask).add_margin(self.margin)
        return bbox, threshold

    def __call__(self, data: Dict) -> Dict:
        logging.info(f"LoadROId: read {data[self.image_key]}")
        image_obj: OpenSlide = self.image_reader.read(data[self.image_key])

        logging.info("LoadROId: get bbox")
        level0_bbox, threshold = self._get_bounding_box(image_obj)
        logging.info(f"LoadROId: level0_bbox: {level0_bbox}")

        # OpenSlide takes absolute location coordinates in the level 0 reference frame,
        # but relative region size in pixels at the chosen level
        scale = image_obj.level_downsamples[self.level]
        scaled_bbox = level0_bbox / scale
        # Monai image_reader.get_data old bug: order of location/size arguments is reversed
        origin = (level0_bbox.y, level0_bbox.x)
        get_data_kwargs = dict(location=origin,
                               size=(scaled_bbox.h, scaled_bbox.w),
                               level=self.level)

        img_data, _ = self.image_reader.get_data(image_obj, **get_data_kwargs)  # type: ignore
        logging.info(f"img_data: {img_data.dtype} {img_data.shape}")
        data[self.image_key] = img_data
        data.update(get_data_kwargs)
        data["origin"] = origin
        data["scale"] = scale
        data["foreground_threshold"] = threshold

        image_obj.close()
        return data
#%%
def is_already_processed(output_tiles_dir):
    if not output_tiles_dir.exists():
        return False

    if len(list(output_tiles_dir.glob("*.png"))) == 0:
        return False

    dataset_csv_path = output_tiles_dir / "dataset.csv"
    try:
        df = pd.read_csv(dataset_csv_path)
    except:
        return False

    return len(df) > 0
#%%
def save_thumbnail(slide_path, output_path, size_target=1024):
    with OpenSlide(str(slide_path)) as openslide_obj:
        scale = size_target / max(openslide_obj.dimensions)
        thumbnail = openslide_obj.get_thumbnail([int(m * scale) for m in openslide_obj.dimensions])
        thumbnail.save(output_path)
        logging.info(f"Saving thumbnail {output_path}, shape {thumbnail.size}")
#%%
def select_tiles(foreground_mask: np.ndarray, occupancy_threshold: float) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Exclude tiles that are mostly background based on estimated occupancy.

    :param foreground_mask: Boolean array of shape (*, H, W).
    :param occupancy_threshold: Tiles with lower occupancy (between 0 and 1) will be discarded.
    :return: A tuple containing which tiles were selected and the estimated occupancies. These will
    be boolean and float arrays of shape (*,), or scalars if `foreground_mask` is a single tile.
    """
    if occupancy_threshold < 0. or occupancy_threshold > 1.:
        raise ValueError("Tile occupancy threshold must be between 0 and 1")
    occupancy = foreground_mask.mean(axis=(-2, -1), dtype=np.float16)
    return (occupancy > occupancy_threshold).squeeze(), occupancy.squeeze()  # type: ignore
#%%
def get_tile_descriptor(tile_location: Sequence[int]) -> str:
    """Format the XY tile coordinates into a tile descriptor."""
    return f"{tile_location[0]:05d}x_{tile_location[1]:05d}y"
#%%
def get_tile_id(slide_id: str, tile_location: Sequence[int]) -> str:
    """Format the slide ID and XY tile coordinates into a unique tile ID."""
    return f"{slide_id}.{get_tile_descriptor(tile_location)}"
#%%
def get_tile_info(sample: Dict["SlideKey", Any], occupancy: float, tile_location: Sequence[int],
                  rel_slide_dir: Path) -> Dict["TileKey", Any]:
    """Map slide information and tiling outputs into tile-specific information dictionary.

    :param sample: Slide dictionary.
    :param occupancy: Estimated tile foreground occuppancy.
    :param tile_location: Tile XY coordinates.
    :param rel_slide_dir: Directory where tiles are saved, relative to dataset root.
    :return: Tile information dictionary.
    """
    slide_id = sample["slide_id"]
    descriptor = get_tile_descriptor(tile_location)
    rel_image_path = f"{rel_slide_dir}/{descriptor}.png"

    tile_info = {
        "slide_id": slide_id,
        "tile_id": get_tile_id(slide_id, tile_location),
        "image": rel_image_path,
        "label": sample.get("label", None),
        "tile_x": tile_location[0],
        "tile_y": tile_location[1],
        "occupancy": occupancy,
        "metadata": {"slide_" + key: value for key, value in sample["metadata"].items()}
    }

    return tile_info
#%%
def get_1d_padding(length: int, tile_size: int) -> Tuple[int, int]:
    """Computes symmetric padding for `length` to be divisible by `tile_size`."""
    pad = (tile_size - length % tile_size) % tile_size
    return (pad // 2, pad - pad // 2)
#%%
def pad_for_tiling_2d(array: np.ndarray, tile_size: int, channels_first: Optional[bool] = True,
                      **pad_kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Symmetrically pads a 2D `array` such that both dimensions are divisible by `tile_size`.

    :param array: 2D image array.
    :param tile_size: Width/height of each tile in pixels.
    :param channels_first: Whether `array` is in CHW (`True`, default) or HWC (`False`) layout.
    :param pad_kwargs: Keyword arguments to be passed to `np.pad()` (e.g. `constant_values=0`).
    :return: A tuple containing:
        - `padded_array`: Resulting array, in the same CHW/HWC layout as the input.
        - `offset`: XY offset introduced by the padding. Add this to coordinates relative to the
        original array to obtain indices for the padded array.
    """
    height, width = array.shape[1:] if channels_first else array.shape[:-1]
    padding_h = get_1d_padding(height, tile_size)
    padding_w = get_1d_padding(width, tile_size)
    padding = [padding_h, padding_w]
    channels_axis = 0 if channels_first else 2
    padding.insert(channels_axis, (0, 0))  # zero padding on channels axis
    padded_array = np.pad(array, padding, **pad_kwargs)
    offset = (padding_w[0], padding_h[0])
    return padded_array, np.array(offset)
#%%
def tile_array_2d(array: np.ndarray, tile_size: int, channels_first: Optional[bool] = True,
                  **pad_kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Split an image array into square non-overlapping tiles.

    The array will be padded symmetrically if its dimensions are not exact multiples of `tile_size`.

    :param array: Image array.
    :param tile_size: Width/height of each tile in pixels.
    :param pad_kwargs: Keyword arguments to be passed to `np.pad()` (e.g. `constant_values=0`).
    :param channels_first: Whether `array` is in CHW (`True`, default) or HWC (`False`) layout.
    :return: A tuple containing:
        - `tiles`: A batch of tiles in NCHW layout.
        - `coords`: XY coordinates of each tile, in the same order.
    """
    padded_array, (offset_w, offset_h) = pad_for_tiling_2d(array, tile_size, channels_first, **pad_kwargs)
    if channels_first:
        channels, height, width = padded_array.shape
    else:
        height, width, channels = padded_array.shape
    n_tiles_h = height // tile_size
    n_tiles_w = width // tile_size

    if channels_first:
        intermediate_shape = (channels, n_tiles_h, tile_size, n_tiles_w, tile_size)
        axis_order = (1, 3, 0, 2, 4)  # (n_tiles_h, n_tiles_w, channels, tile_size, tile_size)
        output_shape = (n_tiles_h * n_tiles_w, channels, tile_size, tile_size)
    else:
        intermediate_shape = (n_tiles_h, tile_size, n_tiles_w, tile_size, channels)
        axis_order = (0, 2, 1, 3, 4)  # (n_tiles_h, n_tiles_w, tile_size, tile_size, channels)
        output_shape = (n_tiles_h * n_tiles_w, tile_size, tile_size, channels)

    tiles = padded_array.reshape(intermediate_shape)  # Split width and height axes
    tiles = tiles.transpose(axis_order)
    tiles = tiles.reshape(output_shape)  # Flatten tile batch dimension

    # Compute top-left coordinates of every tile, relative to the original array's origin
    coords_h = tile_size * np.arange(n_tiles_h) - offset_h
    coords_w = tile_size * np.arange(n_tiles_w) - offset_w
    # Shape: (n_tiles_h * n_tiles_w, 2)
    coords = np.stack(np.meshgrid(coords_w, coords_h), axis=-1).reshape(-1, 2)

    return tiles, coords
#%%
def get_luminance(slide: np.ndarray) -> np.ndarray:
    """Compute a grayscale version of the input slide.

    :param slide: The RGB image array in (*, C, H, W) format.
    :return: The single-channel luminance array as (*, H, W).
    """
    # TODO: Consider more sophisticated luminance calculation if necessary
    return slide.mean(axis=-3, dtype=np.float16)  # type: ignore
#%%
def segment_foreground(slide: np.ndarray, threshold: Optional[float] = None) \
        -> Tuple[np.ndarray, float]:
    """Segment the given slide by thresholding its luminance.

    :param slide: The RGB image array in (*, C, H, W) format.
    :param threshold: Pixels with luminance below this value will be considered foreground.
    If `None` (default), an optimal threshold will be estimated automatically using Otsu's method.
    :return: A tuple containing the boolean output array in (*, H, W) format and the threshold used.
    """
    luminance = get_luminance(slide)
    if threshold is None:
        threshold = skimage.filters.threshold_otsu(luminance)
    logging.info(f"Otsu threshold from luminance: {threshold}")
    return luminance < threshold, threshold
#%%
def generate_tiles(slide_image: np.ndarray, tile_size: int, foreground_threshold: float,
                   occupancy_threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Split the foreground of an input slide image into tiles.

    :param slide_image: The RGB image array in (C, H, W) format.
    :param tile_size: Lateral dimensions of each tile, in pixels.
    :param foreground_threshold: Luminance threshold (0 to 255) to determine tile occupancy.
    :param occupancy_threshold: Threshold (between 0 and 1) to determine empty tiles to discard.
    :return: A tuple containing the image tiles (N, C, H, W), tile coordinates (N, 2), occupancies
    (N,), and total number of discarded empty tiles.
    """
    image_tiles, tile_locations = tile_array_2d(slide_image, tile_size=tile_size,
                                                constant_values=255)
    logging.info(f"image_tiles.shape: {image_tiles.shape}, dtype: {image_tiles.dtype}")
    logging.info(f"Tiled {slide_image.shape} to {image_tiles.shape}")
    foreground_mask, _ = segment_foreground(image_tiles, foreground_threshold)
    selected, occupancies = select_tiles(foreground_mask, occupancy_threshold)
    n_discarded = (~selected).sum()
    logging.info(f"Percentage tiles discarded: {n_discarded / len(selected) * 100:.2f}")

    # FIXME: this uses too much memory
    # empty_tile_bool_mask = check_empty_tiles(image_tiles)
    # selected = selected & (~empty_tile_bool_mask)
    # n_discarded = (~selected).sum()
    # logging.info(f"Percentage tiles discarded after filtering empty tiles: {n_discarded / len(selected) * 100:.2f}")

    # logging.info(f"Before filtering: min y: {tile_locations[:, 0].min()}, max y: {tile_locations[:, 0].max()}, min x: {tile_locations[:, 1].min()}, max x: {tile_locations[:, 1].max()}")

    image_tiles = image_tiles[selected]
    tile_locations = tile_locations[selected]
    occupancies = occupancies[selected]

    if len(tile_locations) == 0:
        logging.warn("No tiles selected")
    else:
        logging.info(f"After filtering: min y: {tile_locations[:, 0].min()}, max y: {tile_locations[:, 0].max()}, min x: {tile_locations[:, 1].min()}, max x: {tile_locations[:, 1].max()}")

    return image_tiles, tile_locations, occupancies, n_discarded
#%%
def save_image(array_chw: np.ndarray, path: Path) -> PIL.Image:
    """Save an image array in (C, H, W) format to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    array_hwc = np.moveaxis(array_chw, 0, -1).astype(np.uint8).squeeze()
    pil_image = PIL.Image.fromarray(array_hwc)
    pil_image.convert('RGB').save(path)
    return pil_image
#%%
def format_csv_row(tile_info: Dict["TileKey", Any], keys_to_save: Iterable["TileKey"],
                   metadata_keys: Iterable[str]) -> str:
    """Format tile information dictionary as a row to write to a dataset CSV tile.

    :param tile_info: Tile information dictionary.
    :param keys_to_save: Which main keys to include in the row, and in which order.
    :param metadata_keys: Likewise for metadata keys.
    :return: The formatted CSV row.
    """
    tile_slide_metadata = tile_info.pop("metadata")
    fields = [str(tile_info[key]) for key in keys_to_save]
    fields.extend(str(tile_slide_metadata[key]) for key in metadata_keys)
    dataset_row = ','.join(fields)
    return dataset_row
#%%
def visualize_tile_locations(slide_sample, output_path, tile_info_list, tile_size, origin_offset):
    # check slide_image size. should be thumbnail size?
    slide_image = slide_sample["image"]
    downscale_factor = slide_sample["scale"]

    fig, ax = plt.subplots()
    ax.imshow(slide_image.transpose(1, 2, 0))
    rects = []
    for tile_info in tile_info_list:
        # change coordinate to the current level from level-0
        # tile location is in the original image cooridnate, while the slide image is after selecting ROI
        xy = ((tile_info["tile_x"] - origin_offset[0]) / downscale_factor,
              (tile_info["tile_y"] - origin_offset[1]) / downscale_factor)
        rects.append(patches.Rectangle(xy, tile_size, tile_size))
    pc = collections.PatchCollection(rects, match_original=True, alpha=0.5, edgecolor="black")
    pc.set_array(np.array([100] * len(tile_info_list)))
    ax.add_collection(pc)
    fig.savefig(output_path)
    plt.close()
#%%
def process_slide(sample: Dict["SlideKey", Any], level: int, margin: int, tile_size: int,
                  foreground_threshold: Optional[float], occupancy_threshold: float, output_dir: Path,
                  thumbnail_dir: Path,
                  tile_progress: bool = False) -> str:
    """Load and process a slide, saving tile images and information to a CSV file.

    :param sample: Slide information dictionary, returned by the input slide dataset.
    :param level: Magnification level at which to process the slide.
    :param margin: Margin around the foreground bounding box, in pixels at lowest resolution.
    :param tile_size: Lateral dimensions of each tile, in pixels.
    :param foreground_threshold: Luminance threshold (0 to 255) to determine tile occupancy.
    If `None` (default), an optimal threshold will be estimated automatically.
    :param occupancy_threshold: Threshold (between 0 and 1) to determine empty tiles to discard.
    :param output_dir: Root directory for the output dataset; outputs for a single slide will be
    saved inside `output_dir/slide_id/`.
    :param tile_progress: Whether to display a progress bar in the terminal.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    thumbnail_dir.mkdir(parents=True, exist_ok=True)
    slide_metadata: Dict[str, Any] = sample["metadata"]
    keys_to_save = ("slide_id", "tile_id", "image", "label",
                    "tile_x", "tile_y", "occupancy")
    metadata_keys = tuple("slide_" + key for key in slide_metadata)
    csv_columns: Tuple[str, ...] = (*keys_to_save, *metadata_keys)
    # print(csv_columns)
    slide_id: str = sample["slide_id"]
    rel_slide_dir = Path(slide_id)
    output_tiles_dir = output_dir / rel_slide_dir
    logging.info(f">>> Slide dir {output_tiles_dir}")
    if is_already_processed(output_tiles_dir):
        logging.info(f">>> Skipping {output_tiles_dir} - already processed")
        return output_tiles_dir

    else:
        output_tiles_dir.mkdir(parents=True, exist_ok=True)
        dataset_csv_path = output_tiles_dir / "dataset.csv"
        dataset_csv_file = dataset_csv_path.open('w')
        dataset_csv_file.write(','.join(csv_columns) + '\n')  # write CSV header

        n_failed_tiles = 0
        failed_tiles_csv_path = output_tiles_dir / "failed_tiles.csv"
        failed_tiles_file = failed_tiles_csv_path.open('w')
        failed_tiles_file.write('tile_id' + '\n')

        slide_image_path = Path(sample["image"])
        logging.info(f"Loading slide {slide_id} ...\nFile: {slide_image_path}")

        # Somehow it's very slow on Datarbicks
        # hack: copy the slide file to a temporary directory
        tmp_dir = tempfile.TemporaryDirectory()
        tmp_slide_image_path = Path(tmp_dir.name) / slide_image_path.name
        logging.info(f">>> Copying {slide_image_path} to {tmp_slide_image_path}")
        shutil.copy(slide_image_path, tmp_slide_image_path)
        sample["image"] = tmp_slide_image_path
        logging.info(f">>> Finished copying {slide_image_path} to {tmp_slide_image_path}")

        # Save original slide thumbnail
        save_thumbnail(slide_image_path, thumbnail_dir / (slide_image_path.name + "_original.png"))

        loader = LoadROId(WSIReader(backend="OpenSlide"), level=level, margin=margin,
                          foreground_threshold=foreground_threshold)
        sample = loader(sample)  # load 'image' from disk

        # Save ROI thumbnail
        slide_image = sample["image"]
        plt.figure()
        plt.imshow(slide_image.transpose(1, 2, 0))
        plt.savefig(thumbnail_dir / (slide_image_path.name + "_roi.png"))
        plt.close()
        logging.info(f"Saving thumbnail {thumbnail_dir / (slide_image_path.name + '_roi.png')}, shape {slide_image.shape}")

        logging.info(f"Tiling slide {slide_id} ...")
        image_tiles, rel_tile_locations, occupancies, _ = \
            generate_tiles(sample["image"], tile_size,
                           sample["foreground_threshold"],
                           occupancy_threshold)

        # origin in level-0 coordinate
        # location in the current level coordiante
        # tile_locations in level-0 coordinate
        tile_locations = (sample["scale"] * rel_tile_locations
                          + sample["origin"]).astype(int)  # noqa: W503

        n_tiles = image_tiles.shape[0]
        logging.info(f"{n_tiles} tiles found")

        tile_info_list = []

        logging.info(f"Saving tiles for slide {slide_id} ...")
        for i in tqdm(range(n_tiles), f"Tiles ({slide_id[:6]}…)", unit="img", disable=not tile_progress):
            try:
                tile_info = get_tile_info(sample, occupancies[i], tile_locations[i], rel_slide_dir)
                tile_info_list.append(tile_info)

                save_image(image_tiles[i], output_dir / tile_info["image"])
                dataset_row = format_csv_row(tile_info, keys_to_save, metadata_keys)
                dataset_csv_file.write(dataset_row + '\n')
            except Exception as e:
                n_failed_tiles += 1
                descriptor = get_tile_descriptor(tile_locations[i])
                failed_tiles_file.write(descriptor + '\n')
                traceback.print_exc()
                warnings.warn(f"An error occurred while saving tile "
                              f"{get_tile_id(slide_id, tile_locations[i])}: {e}")

        dataset_csv_file.close()
        failed_tiles_file.close()

        # tile location overlay
        visualize_tile_locations(sample, thumbnail_dir / (slide_image_path.name + "_roi_tiles.png"), tile_info_list, tile_size, origin_offset=sample["origin"])

        if n_failed_tiles > 0:
            # TODO what we want to do with slides that have some failed tiles?
            logging.warning(f"{slide_id} is incomplete. {n_failed_tiles} tiles failed.")

        logging.info(f"Finished processing slide {slide_id}")

        return output_tiles_dir
#%%

def tile_one_slide(slide_file:str='', save_dir:str='', level:int=0, tile_size:int=256, tile_dir='tiles', thumbnail_dir='thumbnails', metadata:dict={}):
    """
    This function is used to tile a single slide and save the tiles to a directory.
    -------------------------------------------------------------------------------
    Warnings: pixman 0.38 has a known bug, which produces partial broken images.
    Make sure to use a different version of pixman.
    -------------------------------------------------------------------------------

    Arguments:
    ----------
    slide_file : str
        The path to the slide file.
    save_dir : str
        The directory to save the tiles.
    level : int
        The magnification level to use for tiling. level=0 is the highest magnification level.
    tile_size : int
        The size of the tiles.
    """
    slide_id = os.path.basename(slide_file).split('.')[0]
    # slide_sample = {"image": slide_file, "slide_id": slide_id, "metadata": {'TP53': 1, 'Diagnosis': 'Lung Cancer'}}
    slide_sample = {"image": slide_file, "slide_id": slide_id, "metadata": metadata}

    save_dir = Path(save_dir)
    if save_dir.exists():
        print(f"Warning: Directory {save_dir} already exists. ")

    print(f"Processing slide {slide_file} at level {level} with tile size {tile_size}. Saving to {save_dir}.")

    slide_dir = process_slide(
        slide_sample,
        level=level,
        margin=0,
        tile_size=tile_size,
        foreground_threshold=None,
        occupancy_threshold=0.1,
        output_dir=save_dir / tile_dir,
        thumbnail_dir=save_dir / thumbnail_dir,
        tile_progress=True,
    )

    dataset_csv_path = slide_dir / "dataset.csv"
    dataset_df = pd.read_csv(dataset_csv_path)
    assert len(dataset_df) > 0
    failed_csv_path = slide_dir / "failed_tiles.csv"
    failed_df = pd.read_csv(failed_csv_path)
    assert len(failed_df) == 0

    print(f"Slide {slide_file} has been tiled. {len(dataset_df)} tiles saved to {slide_dir}.")


def tile_slides(paths:list, save_dir:str, level:int=0, tile_size:int=256, tile_dir='tiles', thumbnail_dir='thumbnails', metas:list=[]):
    """
    This function is used to tile a single slide and save the tiles to a directory.
    -------------------------------------------------------------------------------
    Warnings: pixman 0.38 has a known bug, which produces partial broken images.
    Make sure to use a different version of pixman.
    -------------------------------------------------------------------------------

    Arguments:
    ----------
    slide_file : str
        The path to the slide file.
    save_dir : str
        The directory to save the tiles.
    level : int
        The magnification level to use for tiling. level=0 is the highest magnification level.
    tile_size : int
        The size of the tiles.
    """
    # if only metadata is empty or with incompatible length
    if len(paths) != len(metas):
        metas = [{} for _ in range(len(paths))]

    for idx, slide_file in enumerate(paths):
        metadata = metas[idx]

        slide_id = os.path.basename(slide_file).split('.')[0]
        # slide_sample = {"image": slide_file, "slide_id": slide_id, "metadata": {'TP53': 1, 'Diagnosis': 'Lung Cancer'}}
        slide_sample = {"image": slide_file, "slide_id": slide_id, "metadata": metadata}

        save_dir = Path(save_dir)
        if save_dir.exists():
            print(f"Warning: Directory {save_dir} already exists. ")

        print(f"Processing slide {slide_file} at level {level} with tile size {tile_size}. Saving to {save_dir}.")

        slide_dir = process_slide(
            slide_sample,
            level=level,
            margin=0,
            tile_size=tile_size,
            foreground_threshold=None,
            occupancy_threshold=0.1,
            output_dir=save_dir / tile_dir,
            thumbnail_dir=save_dir / thumbnail_dir,
            tile_progress=True,
        )

        dataset_csv_path = slide_dir / "dataset.csv"
        dataset_df = pd.read_csv(dataset_csv_path)
        assert len(dataset_df) > 0
        failed_csv_path = slide_dir / "failed_tiles.csv"
        failed_df = pd.read_csv(failed_csv_path)
        assert len(failed_df) == 0

        print(f"Slide {slide_file} has been tiled. {len(dataset_df)} tiles saved to {slide_dir}.")
