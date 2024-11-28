#!/home/gomosak/cnf/bin/python
import os
import re
from fastai.vision.all import * 
import numpy as np
from path import Path 
import torch 
from torchvision import transforms as T 
from data_loading import load_data
from metrics import seg_accuracy
from loss_functions import CombinedLoss
from model_creation import create_model, create_learner
from visualization import show_segmentation_results


def check_cuda_availability():
    """
    Checks if CUDA (GPU support) is available.
    Prints the availability status.
    """
    print(f"CUDA Available: {torch.cuda.is_available()}")


def read_routes(file_path: str) -> Path:
    """
    Reads the dataset base path from a file.

    Parameters:
    - file_path (str): Path to the file containing the base directory path.

    Returns:
    - Path: The dataset base path as a Path object.
    """
    with open(file_path, 'r') as file:
        return Path(file.read().strip())


def check_files(imgs_path: Path, lbls_path: Path):
    """
    Checks and prints the number of files in the images and masks directories.

    Parameters:
    - imgs_path (Path): Path to the images directory.
    - lbls_path (Path): Path to the masks directory.
    """
    print(f'Number of files - Images: {len(list(imgs_path.iterdir()))}, Masks: {len(list(lbls_path.iterdir()))}')


def load_image_and_mask(img_path: Path, mask_path: Path):
    """
    Loads a sample image and mask for inspection.

    Parameters:
    - img_path (Path): Path to the image file.
    - mask_path (Path): Path to the mask file.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: The loaded image and mask as numpy arrays.
    """
    img = np.load(str(img_path))
    msk = np.load(str(mask_path))
    print(f'Image shape: {img.shape}, Mask shape: {msk.shape}')
    return img, msk


def find_highest_model_number(models_path: str) -> int:
    """
    Finds the highest model number in the specified models directory.

    Parameters:
    - models_path (str): Path to the models directory.

    Returns:
    - int: The highest model number.
    """
    models = [d for d in os.listdir(models_path)]
    numbers = [int(re.search(r'[0-9]+', run).group()) for run in models if re.search(r'[0-9]+', run)]
    return max(numbers) if numbers else 0


def main(
    routes_file: str,
    npy_data_dir: str,
    npy_mask_dir: str,
    img_size: tuple,
    batch_size: int,
    architecture: str,
    in_channels: int,
    classes: int,
    opt_func: callable,
    wd: float,
    epochs: int,
):
    """
    Main function to load data, create a model, train it, and save the results.

    Parameters:
    - routes_file (str): Path to the file containing the base directory path.
    - npy_data_dir (str): Subdirectory name for images.
    - npy_mask_dir (str): Subdirectory name for masks.
    - img_size (tuple): Image size for resizing.
    - batch_size (int): Batch size for training.
    - architecture (str): Model architecture for training.
    - in_channels (int): Number of input channels.
    - classes (int): Number of output classes.
    - opt_func (callable): Optimizer function.
    - wd (float): Weight decay for the optimizer.
    - epochs (int): Number of training epochs.

    Returns:
    - None
    """
    # Check CUDA availability
    check_cuda_availability()

    # Load base paths
    base_path = read_routes(routes_file)
    imgs_path = base_path / npy_data_dir
    lbls_path = base_path / npy_mask_dir

    # Check file counts
    check_files(imgs_path, lbls_path)

    # Load a sample image and mask for inspection
    sample_img_path = sorted(imgs_path.iterdir(), key=lambda x: int(x.stem))[12]
    sample_msk_path = sorted(lbls_path.iterdir(), key=lambda x: int(x.stem))[12]
    load_image_and_mask(sample_img_path, sample_msk_path)


    # Load data
    dls = load_data(imgs_path, lbls_path, img_size, batch_size)

    # Create model
    model = create_model(architecture, in_channels=in_channels, classes=classes)

    # Define loss function
    loss_func = CombinedLoss()

    # Create learner
    learner = create_learner(model, loss_func, opt_func=partial(opt_func, wd=wd), db=dls, metrics=[seg_accuracy])

    # Train the model
    learner.fine_tune(epochs)

    # Save the trained model
    highest_number = find_highest_model_number('models')
    model_path = f'models/model_{highest_number + 1}.pkl'
    learner.export(model_path)
    print(f"Model saved at: {model_path}")


if __name__ == "__main__":
    # Example input arguments
    main(
        routes_file='routes.txt',
        npy_data_dir='npy_data_504',
        npy_mask_dir='npy_mask_504',
        img_size=(544, 480),
        batch_size=1,
        architecture="resnet101",
        in_channels=6,
        classes=7,
        opt_func=SGD,
        wd=0.00039897560969184224,
        epochs=1,
    )
