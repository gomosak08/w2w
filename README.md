

Wall to Wall

Note: All the code provided below is in an experimental phase; therefore, it must be used with caution.

In this section, I will present some images demonstrating both the best and worst results of the models.

Example Showcase:

Our example starts with the image and its corresponding mask, where each pixel in the mask holds the value of its corresponding pixel in the image. The mask represents the classification of the image. Therefore, the objective is to train a model to predict the classification for each pixel in the images.

<img src="https://github.com/CONAFOR-GTMRV/wtow/assets/79944448/e2029eb4-4881-4efa-891b-8dc96a60d1da" width="500" height="300">

Then, we proceed to construct batches to accelerate processing using GPU. For this purpose, I've developed a class to manage images consisting of 6 bands along with their corresponding masks. Each image has a size of 256x256 pixels. Considering the available resources, I create batches containing 4 images. Additionally, I've incorporated some basic transformations such as rotation, vertical flipping, zoom, and brightness adjustment to enhance the completeness of the dataset. However, there is pot 
ential to apply more sophisticated transformations.

<img src="https://github.com/CONAFOR-GTMRV/wtow/assets/79944448/65560abf-1c07-4262-95bd-a1de67f1f3c9" width="400" height="250">


I have experimented with various model architectures, such as ResNet, FPN, and UnetPlusPlus, in order to find the optimal model for the task. To facilitate this process, I utilized the segmentation_models_pytorch library. Additionally, I explored different loss functions, optimizers, and evaluation metrics to assess the performance of the models. Currently, I am employing the MulticlassDiceLoss as the loss function and DiceMulticlass as the metric function. As for optimizers, I opted for a classic one, utilizing the Ranger optimizer.

Here's an example showcasing the comparison between a random prediction (left) and the actual mask image (right):

<img src="https://github.com/gomosak08/wall-to-wall/assets/79944448/797d4aa9-03f6-4f5c-97a2-33a9ed018a96" width="500" height="500">

Best Model Result:

This image showcases the best result achieved by the model, boasting a remarkable general accuracy of .948:

<img src = "https://github.com/gomosak08/wall-to-wall/assets/79944448/87173383-71dd-4062-959b-37065f59d26b" width = "500" heigth = "500">

Other Results:
Here are additional images representing the outcomes of the model:

Image 1:

<img src = "https://github.com/gomosak08/wall-to-wall/assets/79944448/87173383-71dd-4062-959b-37065f59d26b" width = "500" heigth = "500">

Image 2:

<img src = "https://github.com/gomosak08/wall-to-wall/assets/79944448/9f9a31e1-927c-4e27-94ac-a886b4dd5594" width = "500" heigth = "500">



I have also attempted to enhance the dataset by generating virtual data based on the real data. However, this approach did not yield satisfactory results because the virtual data exhibits a normal distribution, whereas the real data does not. Therefore, my current strategy is to improve the real data by applying transformations.

We conducted an exercise in the state of Chihuahua using segmented data developed by GTMRSV through CONAFOR. Our preliminary results showed promising patterns; however, the data requires further post-processing to improve its clarity and accuracy. Specifically, one key adjustment will involve filtering out areas smaller than 1 hectare, which can help reduce noise and enhance the precision of the visual output. This step is essential for focusing on meaningful spatial patterns and ensuring that the final image provides a clear and actionable representation of the landscape.


<img src = "https://github.com/user-attachments/assets/cecaa50b-c4d2-48e8-b2f7-62073ffa8da8" width = "500" heigth = "500">


