# Wall to Wall

**Author:** Kevin Saúl Gómez Molina  
**Email:** [gomosak@outlook.es](mailto:gomosak@outlook.es)  

If you have any questions or need further assistance with this project, feel free to reach out via any of the contact options above.

---
# Experimental Results and Analysis

## Table of Contents
1. [Introduction](#introduction)
2. [Setup](#setup)
3. [Training](#training)
4. [Inference](#inference)
5. [Split images](#spliting-images)
6. [Example Showcase](#example-showcase)
7. [Best Model Result](#best-model-result)
8. [Other Results](#other-results)
9. [Last Result in Chihuahua](#last-result-in-chihuahua)

---

## Introduction

**Note:** All the code provided below is in an experimental phase; therefore, it must be used with caution.

The primary objective of this project is to develop accurate and reliable land forest coverage maps using advanced image segmentation techniques. By classifying each pixel within an image, we aim to produce detailed thematic maps that can support environmental analysis and decision-making processes. The project is focused on utilizing state-of-the-art deep learning architectures for image segmentation, leveraging frameworks such as **PyTorch** and the **segmentation_models_pytorch** library.

A key challenge in this work is the limited availability of high-quality training masks, which affects the ability of the models to generalize effectively. Additionally, the computational cost of training these models is significant due to the complexity of the data and the need for pixel-level precision. Another limitation arises from the imbalance in the data distribution: most of the dataset represents forested areas, while only a small fraction corresponds to urban regions or water bodies. This imbalance creates challenges for the model in accurately identifying underrepresented classes.

The current work is centered on the state of **Chihuahua**, utilizing geospatial data to create accurate land coverage maps. We are employing **accuracy** as the primary evaluation metric to assess the performance of the models. By addressing these challenges and refining the methodology, this project aims to contribute to the development of robust tools for environmental monitoring and land management.

In this document, we will showcase experimental results, presenting both the best and worst outcomes of the models. The examples highlight the strengths and limitations of the approach, offering insights into areas that require further improvement.


---

## Setup
For a specific instance like the AWS G5, you can run `drivers.sh`, which creates the environment and installs all the necessary drivers, including the CUDA toolkit. After running this script, you need to reboot the system and then execute `setup.sh` to install all the required libraries.
The setup process for this project is tailored specifically for Ubuntu systems. To streamline the installation, we provide clear instructions for installing PyTorch and other required dependencies. The process ensures all necessary tools, libraries, and the virtual environment are properly configured. Follow the steps below to set up the environment:


1. **Creating a Virtual Environment**:
   - A Python virtual environment is created to isolate the project dependencies from the system Python environment, ensuring a clean and manageable setup:
     ```bash
     python3 -m venv path/to/venv
     source path/to/venv/bin/activate
2. **PyTorch and FastAI Installation**:
   - The project requires **PyTorch** and **FastAI 2.7.18**. To ensure compatibility, we recommend following the official installation guides:
     - [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
     - [FastAI Installation Guide](https://docs.fast.ai/)

      Use the following command to install these versions directly:
      ```bash
      pip install fastai==2.7.18 
      ```
    - Ensure that you select the appropriate CUDA version for your system during the installation of PyTorch. You can find compatible options in the PyTorch guide.

    - To verify that PyTorch is installed correctly, run the following commands in your terminal:
      ```bash
      ipython
      ```
      Then, inside the IPython shell:
      ```bash
      import torch
      print(torch.__version__)
      ```
      If PyTorch is installed correctly, this will return the version of PyTorch installed on your system.

3. **Installing Dependencies**:
   - After activating the virtual environment, all required Python packages are installed from the `requirements.txt` file:
     ```bash
     pip install -r requirements.txt
     ```

### Notes:
- Ensure you have Python 3.10 or later installed on your system before running the setup script.
- If any issues arise during the installation of PyTorch or FastAI, refer to the official documentation linked above for troubleshooting.




---

## Training

`Model.py` is a Python script designed to train a machine learning model with customizable parameters. You can modify the routes and other parameters in the script to suit your specific dataset and training requirements. Below is a detailed guide on how to configure each parameter for your personal case.

---

#### Overview

The script processes input data, applies transformations, and trains a neural network using the specified architecture and hyperparameters. Key parameters that you need to configure include the routes for input data and masks, image dimensions, model architecture, optimization function, and training settings.

---

#### Modifying Parameters

Below is a detailed explanation of each parameter in the `main()` function and how you should modify it for your case:

| Parameter        | Description                                                                         |
|------------------|-------------------------------------------------------------------------------------|
| `routes_file`    | Path to the file containing routing information.                                   |
| `npy_data_dir`   | Directory containing `.npy` files for input data.                                  |
| `npy_mask_dir`   | Directory containing `.npy` files for mask data.                                   |
| `img_size`       | Tuple specifying the image size (width, height).                                   |
| `batch_size`     | Number of samples per batch during training.                                       |
| `architecture`   | Neural network architecture to use (e.g., `resnet101`).                            |
| `in_channels`    | Number of input channels for the model.                                            |
| `classes`        | Number of output classes for classification or segmentation.                       |
| `opt_func`       | Optimization function (e.g., `SGD`, `Adam`).                                       |
| `wd`             | Weight decay value for regularization.                                             |
| `epochs`         | Number of training epochs.                                                         |
-------



### Inference

This script is designed to predict segmentation masks using a trained machine learning model and save the results as `.npy` files. Follow the steps below to execute the script.

---

 **Input Files**:
   - **`routes.txt`**: A text file containing the base directory path for your dataset.
   - **Pre-trained Model**: Path to the `.pkl` file containing the trained model (e.g., `models/model.pkl`).
   - **Images Directory**: The directory specified in `routes.txt` should contain `.npy` files of input images.

---

#### Steps to Run the Script

##### 1. Prepare Input Files and Folders
   - Create a `routes.txt` file containing the base path to your dataset. For example:
     ```
     /path/to/dataset
     ```
   - Place `.npy` files for images inside a subfolder (e.g., `/path/to/dataset/data`).

##### 2. Verify Parameters in the Script
   Update the following variables in the script as needed:
   - `routes_file`: Path to the `routes.txt` file (default: `'routes.txt'`).
   - `learner_path`: Path to the pre-trained `.pkl` model (default: `'models/model.pkl'`).
   - `preds_dir`: Directory to save predicted masks (default: `'preds/'`).
   - `limit`: Maximum number of images to process (default: `2`).

##### 3. Run the Script
Execute the script in your terminal or Python environment:

```bash
python inference.py
```
#### How the Script Works

1. **Load Dataset Path**:
   - The script reads the base directory path from `routes.txt` and constructs the input images directory path (e.g., `data`).

2. **Load Pre-trained Model**:
   - It loads the pre-trained model specified in `learner_path` for segmentation predictions.

3. **Predict Masks**:
   - The script iterates through the `.npy` files in the input directory, predicts segmentation masks for each file, and extracts the final mask using the model.

4. **Save Predicted Masks**:
   - The predicted masks are saved as `.npy` files in the specified `preds_dir`.

---

#### Outputs

After running the script:
1. **Predicted Masks**:
   - The masks will be saved in the `preds_dir` folder with filenames matching the input files' names.

2. **Logs**:
   - Progress will be displayed in the terminal, including a progress bar for processed files.

---

#### Example Configuration

#### Input Setup
- `routes.txt`:
  /home/user/dataset
- Model Path: `models/model_7.pkl`
- Images Directory: `/home/user/dataset/npy_data_504`

##### Script Execution
Run the following command:
```bash
python Model.py
```
Expected output:
- Predicted masks saved in the preds/ directory.

##### Notes
- Modify limit to process more images by changing its value in the predict_and_save_masks function call.
Ensure your .npy images are properly preprocessed and compatible with the model.
For large datasets, consider running the script on a machine with sufficient memory and GPU support.

------
### Spliting images

#### Instructions to Run the Raster Image and Mask Processing Script

This script processes raster images and masks, divides them into smaller sub-images and sub-masks, and saves the results. Follow the guide below to configure and execute the script.

---

##### Overview

The script:
1. Loads raster images and masks as NumPy arrays.
2. Preprocesses the mask to map specific values to target values.
3. Cuts borders to make the dimensions divisible by the target size.
4. Divides the images and masks into smaller sub-images and sub-masks.
5. Saves the processed images and masks into specified directories.

---

#### Script Parameters

| Parameter        | Description                                                                          |
|-------------------|--------------------------------------------------------------------------------------|
| `mask_path`       | Path to the mask raster file.                                                       |
| `data_path`       | Path to the data raster file.                                                       |
| `image_dir`       | Directory to save image pieces.                                                     |
| `mask_dir`        | Directory to save mask pieces.                                                      |
| `target_width`    | Width of each sub-image.                                                            |
| `target_height`   | Height of each sub-image.                                                           |

---

#### Example Configuration

Here is how you can configure the script to process your raster data:

- **Input Files**:
  - Mask: `mask.tif`
  - Data: `data.tif`

- **Directories**:
  - Save images in the `data/` directory.
  - Save masks in the `mask/` directory.

- **Sub-image Dimensions**:
  - Width: `544`
  - Height: `480`

---

#### Running the Script

1. **Prepare Input Files**:
   - Place the mask file and data file in the same directory as the script or provide their full paths.

2. **Run the Script**:
   Execute the script with:
   ```bash
   python split_image.py
3. **Check the Output**:
    - Processed images will be saved in the specified image_dir (e.g., data/).
    - Processed masks will be saved in the specified mask_dir (e.g., mask/).

### How the Script Works

##### 1. **Load Raster Files**
   - The script reads the raster data (image and mask) using the `load_raster_as_array()` function and converts them into NumPy arrays for further processing.

##### 2. **Preprocess the Mask**
   - The `preprocess_mask()` function remaps specific values in the mask to target values based on a predefined mapping. For example:
     - Values like `-5` are mapped to `0`.
     - Values like `2`, `3`, `6`, etc., are mapped to `1`.
   - This step ensures that the mask values are consistent and ready for segmentation.

##### 3. **Cut Borders**
   - The `cut_borders()` function trims the image and mask borders to make their dimensions divisible by the specified sub-image size (`target_width` and `target_height`).
   - This step ensures that the images can be evenly divided into sub-images without leaving partial tiles.

##### 4. **Divide into Sub-images and Sub-masks**
   - The `divide_image_np()` function splits the raster image into smaller sub-images based on the number of rows and columns calculated by `get_values()`.
   - The `divide_image_masks()` function similarly divides the mask into corresponding sub-masks.

##### 5. **Save Results**
   - The `save_images_and_masks()` function saves the sub-images and sub-masks as `.npy` files in the specified directories (`image_dir` and `mask_dir`).
   - Masks with all zero values are skipped to save storage space.

#### Notes

- **Custom Mask Preprocessing**:
  - You can modify the `value_map` in the `preprocess_mask()` function to adapt the mapping of mask values to your specific use case.

- **Empty Masks**:
  - Sub-masks with all zero values are skipped during the saving process to optimize storage.

- **Performance Considerations**:
  - The script is optimized for processing large raster images. Ensure that your system has enough memory to handle large arrays, especially for high-resolution datasets.

- **Dependencies**:
  - Ensure that the required libraries (`rasterio`, `numpy`) are installed in your Python environment.

---

### Outputs

After running the script, the following outputs will be generated:

1. **Processed Sub-images**:
   - Saved in the specified `image_dir` directory as `.npy` files.
   - Example filenames:
     - `i/0.npy`
     - `i/1.npy`
     - ...

2. **Processed Sub-masks**:
   - Saved in the specified `mask_dir` directory as `.npy` files.
   - Example filenames:
     - `m/0.npy`
     - `m/1.npy`
     - ...


















## Example Showcase

Our example starts with the image and its corresponding mask, where each pixel in the mask holds the value of its corresponding pixel in the image. The mask represents the classification of the image. Therefore, the objective is to train a model to predict the classification for each pixel in the images.

<img src="image/imagen1.png" width="500" height="300">

Then, we proceed to construct batches to accelerate processing using GPU. For this purpose, I've developed a class to manage images consisting of 6 bands along with their corresponding masks. Each image has a size of 256x256 pixels. Considering the available resources, We create batches containing 4 images. Additionally, I've incorporated some basic transformations such as rotation, vertical flipping, zoom, and brightness adjustment to enhance the completeness of the dataset. However, there is potential to apply more sophisticated transformations.

<img src="image/imagen2.png" width="400" height="250">

We have experimented with various model architectures, such as ResNet, FPN, and UnetPlusPlus, in order to find the optimal model for the task. To facilitate this process, We utilized the segmentation_models_pytorch library. Additionally, We explored different loss functions, optimizers, and evaluation metrics to assess the performance of the models. Currently, We are employing the MulticlassDiceLoss as the loss function and DiceMulticlass as the metric function. As for optimizers, We opted for a classic one, utilizing the Ranger optimizer.

Here's an example showcasing the comparison between a random prediction (left) and the actual mask image (right):

<img src="image/imagen3.png" width="500" height="500">

---

## Best Model Result

This image showcases the best result achieved by the model, boasting a remarkable general accuracy of .948:

<img src="image/imagen4.png" width="500" height="500">

---

## Other Results

Here are additional images representing the outcomes of the model:

**Image 1:**

<img src="image/imagen5.png" width="500" height="500">

**Image 2:**

<img src="image/imagen6.png" width="500" height="500">

We have also attempted to enhance the dataset by generating virtual data based on the real data. However, this approach did not yield satisfactory results because the virtual data exhibits a normal distribution, whereas the real data does not. Therefore, my current strategy is to improve the real data by applying transformations.

---

## Last Result in Chihuahua

We conducted an exercise in the state of Chihuahua using segmented data developed by GTMRSV through CONAFOR. Our preliminary results showed promising patterns; however, the data requires further post-processing to improve its clarity and accuracy. Specifically, one key adjustment will involve filtering out areas smaller than 1 hectare, which can help reduce noise and enhance the precision of the visual output. This step is essential for focusing on meaningful spatial patterns and ensuring that the final image provides a clear and actionable representation of the landscape. This exercise achieved an accuracy of 87% in the initial mode


<img src = "image/imagen7.png" width = "500" heigth = "500">
