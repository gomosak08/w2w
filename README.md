# Wall to Wall

**Author:** Kevin Saúl Gómez Molina  
**Email:** [gomosak@outlook.es](mailto:gomosak@outlook.es)  

If you have any questions or need further assistance with this project, feel free to reach out via any of the contact options above.

---
# Experimental Results and Analysis

## Table of Contents
1. [Introduction](#introduction)
2. [Setup](#setup)
3. [Example Showcase](#example-showcase)
4. [Best Model Result](#best-model-result)
5. [Other Results](#other-results)
6. [Last Result in Chihuahua](#last-result-in-chihuahua)

---

## Introduction

**Note:** All the code provided below is in an experimental phase; therefore, it must be used with caution.

The primary objective of this project is to develop accurate and reliable land forest coverage maps using advanced image segmentation techniques. By classifying each pixel within an image, we aim to produce detailed thematic maps that can support environmental analysis and decision-making processes. The project is focused on utilizing state-of-the-art deep learning architectures for image segmentation, leveraging frameworks such as **PyTorch** and the **segmentation_models_pytorch** library.

A key challenge in this work is the limited availability of high-quality training masks, which affects the ability of the models to generalize effectively. Additionally, the computational cost of training these models is significant due to the complexity of the data and the need for pixel-level precision. Another limitation arises from the imbalance in the data distribution: most of the dataset represents forested areas, while only a small fraction corresponds to urban regions or water bodies. This imbalance creates challenges for the model in accurately identifying underrepresented classes.

The current work is centered on the state of **Chihuahua**, utilizing geospatial data to create accurate land coverage maps. We are employing **accuracy** as the primary evaluation metric to assess the performance of the models. By addressing these challenges and refining the methodology, this project aims to contribute to the development of robust tools for environmental monitoring and land management.

In this document, we will showcase experimental results, presenting both the best and worst outcomes of the models. The examples highlight the strengths and limitations of the approach, offering insights into areas that require further improvement.


---

## Setup
For a specific instance like the AWS G5, you can run ```setup.sh```, which will create the environment, install all the necessary drivers, the CUDA toolkit, and all the required libraries.

The setup process for this project is tailored specifically for Ubuntu systems. To streamline the installation, we provide clear instructions for installing PyTorch and other required dependencies. The process ensures all necessary tools, libraries, and the virtual environment are properly configured. Follow the steps below to set up the environment:

1. **PyTorch and FastAI Installation**:
   - The project requires **PyTorch 2.0** and **FastAI 2.7.12**. To ensure compatibility, we recommend following the official installation guides:
     - [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
     - [FastAI Installation Guide](https://docs.fast.ai/)

   Use the following command to install these versions directly:
   ```bash
   fastai==2.7.18
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

2. **Creating a Virtual Environment**:
   - A Python virtual environment is created to isolate the project dependencies from the system Python environment, ensuring a clean and manageable setup:
     ```bash
     python3 -m venv path/to/venv
     source path/to/venv/bin/activate
     ```

3. **Installing Dependencies**:
   - After activating the virtual environment, all required Python packages are installed from the `requirements.txt` file:
     ```bash
     pip install -r requirements.txt
     ```

### Notes:
- Ensure you have Python 3.10 or later installed on your system before running the setup script.
- If any issues arise during the installation of PyTorch or FastAI, refer to the official documentation linked above for troubleshooting.




---

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