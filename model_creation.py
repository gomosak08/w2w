import segmentation_models_pytorch as smp
from fastai.vision.all import *

def create_model(encoder_name,in_channels,classes):
    """
    Creates a U-Net model using the segmentation models library.

    Parameters:
    - `encoder_name` (str): The name of the encoder to use for the U-Net model.
    - `in_channels` (int): The number of input channels for the model.
    - `classes` (int): The number of output classes for the model.

    Returns:
    - `torch.nn.Module`: The created U-Net model moved to the GPU.
    """
    return smp.Unet(encoder_name=encoder_name ,in_channels=in_channels, classes=classes).to('cuda')


    
def create_learner(model,loss_func,opt_func,db,metrics):
    """
    Creates a fastai Learner object for training.

    Parameters:
    - `model` (torch.nn.Module): The model to be trained.
    - `loss_func` (callable): The loss function to use during training.
    - `opt_func` (callable): The optimizer function to use during training.
    - `db` (DataLoaders): The dataloaders for training and validation data.
    - `metrics` (list of callables): The list of metrics to evaluate during training.

    Returns:
    - `Learner`: The fastai Learner object.
    """
    model = model
    opt_func = opt_func
    learn = Learner(db, model, loss_func=loss_func, opt_func= opt_func, metrics=metrics)
    return learn