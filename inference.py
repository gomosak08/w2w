from fastai.vision.all import *  # FastAI library for deep learning tasks
from image_processing import MSTensorImage  # Custom image processing module for handling .npy images
import numpy as np  # Library for numerical operations
from pathlib import Path  # For handling file paths
from tqdm import tqdm



def load_paths_from_file(file_path: str) -> Path:
    """
    Reads a file containing a directory path and returns it as a Path object.

    Parameters:
    - file_path (str): Path to the text file containing the directory path.

    Returns:
    - Path: The directory path as a Path object.
    """
    with open(file_path, 'r') as file:
        path_str = file.read().strip()  # Read and strip any extra whitespace
    return Path(path_str)


def predict_and_save_masks(learner_path: str, imgs_dir: Path, preds_dir: str, limit: int = 2) -> None:
    """
    Predicts segmentation masks for a set of images and saves the results as .npy files.

    Parameters:
    - learner_path (str): Path to the .pkl file containing the trained learner.
    - imgs_dir (Path): Directory containing the input images as .npy files.
    - preds_dir (str): Directory to save the predicted masks.
    - limit (int): Maximum number of images to process (default: 2).

    Returns:
    - None
    """
    # Load the pre-trained learner
    learner = load_learner(learner_path)

    # Create output directory if it doesn't exist
    preds_path = Path(preds_dir)
    preds_path.mkdir(parents=True, exist_ok=True)

    # Sort files in the input directory by their numeric stem
    img_path_dir = sorted([f for f in imgs_dir.iterdir() if f.is_file()], key=lambda x: int(x.stem))

    # Iterate through the images, up to the specified limit
    for img_file in tqdm(img_path_dir[:limit], total=len(img_path_dir[:limit])):

        # Load the image as a MSTensorImage
        img = MSTensorImage.create(img_file)

        # Make predictions using the learner
        predicted_mask, _, _ = learner.predict(img)

        # Get the final segmentation mask
        final_mask = predicted_mask.argmax(dim=0)

        # Save the mask as a .npy file
        np.save(preds_path / f'{img_file.stem}.npy', final_mask)



# File containing the base path to the dataset
routes_file = 'routes.txt'

# Model and data directories
learner_path = 'models/model.pkl'
preds_dir = 'preds/'

# Load paths from the routes file
base_path = load_paths_from_file(routes_file)

# Define input image directory
imgs_dir = base_path / 'data'

# Predict masks and save results
predict_and_save_masks(learner_path, imgs_dir, preds_dir, limit=2)



