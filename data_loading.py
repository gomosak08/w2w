from fastai.vision.all import DataBlock,TransformBlock
import fastai.vision.all as fv
from functools import partial
from image_processing import MSTensorImage, open_npy_mask

from fastai.vision.core import TensorMask


def load_data(path, mask_path, img_size, batch_size):
    """
    Loads and preprocesses data for training and validation, including image and mask data.

    Parameters:
    - `path` (str): The directory path where the dataset is located.
    - `mask_path` (str): The directory path where the mask data is located.
    - `img_size` (int): The size to which images should be resized.
    - `batch_size` (int): The number of samples per batch to load.

    Returns:
    - `DataLoaders`: A fastai `DataLoaders` object containing the training and validation dataloaders.
    """
    tfms = fv.aug_transforms(size=img_size,
                             max_rotate=30,
                             max_zoom=2.0,
                             max_lighting=0.2,  # Random changes in lighting conditions
                             flip_vert=True,
                             mult=2,
                             xtra_tfms=[fv.RandomErasing(p=0.5)]
                            )

    dblock = DataBlock(
        blocks=(TransformBlock(type_tfms=partial(MSTensorImage.create, chnls_first=True)),
                TransformBlock(type_tfms=partial(open_npy_mask, cls=TensorMask, path=mask_path + '/'))),
        get_items=fv.get_files,
        splitter=fv.RandomSplitter(valid_pct=0.1),
        batch_tfms=tfms,
    )

    return dblock.dataloaders(path, bs=batch_size)
