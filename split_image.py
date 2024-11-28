import numpy as np
from PIL import Image as im
from matplotlib import pyplot as plt
import rasterio


def load_raster_as_array(file_path):
    """
    Loads a raster file into a NumPy array.

    Args:
    - file_path (str): Path to the raster file.

    Returns:
    - np.ndarray: The raster data as a NumPy array.
    """
    with rasterio.open(file_path) as src:
        return src.read()


def preprocess_mask(mask):
    """
    Preprocesses a mask by applying transformations to specific values.

    Args:
    - mask (np.ndarray): The mask as a NumPy array.

    Returns:
    - np.ndarray: The preprocessed mask.
    """
    mask = np.squeeze(mask)
    value_map = {
        -5: 0, 2: 1, 3: 1, 6: 1, 12: 1, 28: 3, 29: 2,
        30: 5, 31: 6, 32: 4, 280: 3, 14: 1, 21: 3,
        23: 3, 25: 3, 26: 3, 27: 3, 290: 2
    }
    for old_value, new_value in value_map.items():
        mask[mask == old_value] = new_value
    return mask


def cut_borders(array, border_x, border_y):
    """
    Cuts borders from a 2D or 3D array.

    Args:
    - array (np.ndarray): The array to cut.
    - border_x (tuple): Number of pixels to cut from top and bottom.
    - border_y (tuple): Number of pixels to cut from left and right.

    Returns:
    - np.ndarray: The cropped array.
    """
    return array[..., border_x[0]:-border_x[1], border_y[0]:-border_y[1]]


def divide_image_np(image, rows, cols):
    """
    Divides a 3D image into smaller sub-images.

    Args:
    - image (np.ndarray): The image as a 3D NumPy array (channels, height, width).
    - rows (int): Number of rows to divide the image into.
    - cols (int): Number of columns to divide the image into.

    Returns:
    - np.ndarray: Array of sub-images.
    """
    h, w = image[0].shape
    return np.array([
        image[:, h//rows*row:h//rows*(row+1), w//cols*col:w//cols*(col+1)]
        for row in range(rows) for col in range(cols)
    ])


def divide_image_masks(mask, rows, cols):
    """
    Divides a 2D mask into smaller sub-masks.

    Args:
    - mask (np.ndarray): The mask as a 2D NumPy array (height, width).
    - rows (int): Number of rows to divide the mask into.
    - cols (int): Number of columns to divide the mask into.

    Returns:
    - np.ndarray: Array of sub-masks.
    """
    h, w = mask.shape
    return np.array([
        mask[h//rows*row:h//rows*(row+1), w//cols*col:w//cols*(col+1)]
        for row in range(rows) for col in range(cols)
    ])


def save_images_and_masks(images, masks, image_dir, mask_dir):
    """
    Saves images and masks to specified directories.

    Args:
    - images (np.ndarray): Array of images.
    - masks (np.ndarray): Array of masks.
    - image_dir (str): Directory to save images.
    - mask_dir (str): Directory to save masks.
    """
    for i, mask in enumerate(masks):
        if not np.all(mask == 0):  # Skip empty masks
            mask_array = mask.astype(np.float32)
            np.save(f'{mask_dir}/{i}.npy', mask_array)
            
            img_array = images[i].astype(np.float32)
            np.save(f'{image_dir}/{i}.npy', img_array)

def get_values(xy, px, py):
    """
    Calculates the differences and number of rows/columns for dividing an image.

    Args:
    - x (int): Width of the image.
    - y (int): Height of the image.
    - px (int): Desired width of each sub-image.
    - py (int): Desired height of each sub-image.

    Returns:
    - tuple: 
        - (float) Horizontal difference to evenly distribute sub-images.
        - (float) Vertical difference to evenly distribute sub-images.
        - (int) Number of rows the image can be divided into.
        - (int) Number of columns the image can be divided into.
    """
    _,x,y = xy

    rows = x // px  # Number of rows
    cols = y // py  # Number of columns

    diffx = x - (px * rows)  # Remaining horizontal pixels
    diffy = y - (py * cols)  # Remaining vertical pixels

    if diffx//2%2!=0:
        x = (diffx//2+1,diffx//2 +1)

    else:
        x = (diffx//2,diffx//2)


    if diffy//2%2!=0:
        y = (diffy//2,diffy//2 +1)

    else:
        y = (diffy//2,diffy//2)


    return x, y, rows, cols


def main(mask_path, data_path, image_dir, mask_dir, target_width, target_height):
    """
    Main function to process raster images and masks, divide them into sub-images/masks, and save results.
    
    Args:
        mask_path (str): Path to the mask raster file.
        data_path (str): Path to the data raster file.
        image_dir (str): Directory to save image pieces.
        mask_dir (str): Directory to save mask pieces.
        target_width (int): Width of each sub-image.
        target_height (int): Height of each sub-image.
    """
    # Load raster files
    mask_array = load_raster_as_array(mask_path)
    image_array = load_raster_as_array(data_path)

    # Preprocess mask
    mask_array = preprocess_mask(mask_array)

    # Replace NaN values in image array with 0
    image_array = np.nan_to_num(image_array)

    # Cut borders
    shape = image_array.shape
    x, y, r, c = get_values(shape, target_width, target_height)
    cut_mask = cut_borders(mask_array, x, y)
    cut_image = cut_borders(image_array, x, y)

    # Divide images and masks into smaller pieces
    sub_images = divide_image_np(cut_image, r, c)
    sub_masks = divide_image_masks(cut_mask, r, c)

    # Save images and masks
    save_images_and_masks(sub_images, sub_masks, image_dir, mask_dir)
    print(f"Images saved in {image_dir}\nMasks saved in {mask_dir}")


if __name__ == "__main__":
    # Define paths, directories, and image dimensions
    mask_path = "mascara320.tif"
    data_path = "data320.tif"
    image_dir = "i"
    mask_dir = "m"
    target_width = 544
    target_height = 480

    # Call the main function with parameters
    main(mask_path, data_path, image_dir, mask_dir, target_width, target_height)
