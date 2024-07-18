import matplotlib.pyplot as plt

def show_segmentation_results(learn, nrows=1, ncols=3):
    """
    Displays segmentation results including the input image, true mask, and predicted mask.

    Parameters:
    - `learn` (Learner): The fastai Learner object containing the trained model and data.
    - `nrows` (int, optional): Number of rows of images to display. Default is `1`.
    - `ncols` (int, optional): Number of columns of images to display. Default is `3`.

    The function shows a batch of data from the dataloader, and for each image, it displays:
    1. The input image
    2. The true mask
    3. The predicted mask
    """
    
    # Get a batch of data
    xb, yb = learn.dls.one_batch() #to('cpu')
    # Get the model's predictions
    preds = learn.model(xb)

    yb = yb.to('cpu')
    xb = xb.to('cpu')
    
    for i in range(nrows):
        fig, axes = plt.subplots(1, ncols, figsize=(15, 5))

        # Show the input image
        img = xb[i][:3,:,:].permute(1, 2, 0)  # Rearrange the dimensions to (height, width, channels)
        axes[0].imshow(img*3.0)
        axes[0].set_title("Input Image")

        # Show the true mask
        true_mask = yb[i]
        axes[1].imshow(true_mask, cmap='inferno',vmax = 7, vmin = 0)
        axes[1].set_title("True Mask")

        # Show the predicted mask
        pred_mask = preds[i].argmax(dim=0).to('cpu')  # Get the channel with the highest probability
        axes[2].imshow(pred_mask, cmap='inferno',vmax = 7, vmin = 0)
        axes[2].set_title("Predicted Mask")

        for ax in axes:
            ax.axis('off')

    plt.show()