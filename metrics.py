import fastai.vision.all as fv

def seg_accuracy(yp, y):
    """
    Calculates segmentation accuracy.

    Parameters:
    - `yp` (Tensor): Predicted tensor.
    - `y` (Tensor): Ground truth tensor.

    Returns:
    - `float`: The accuracy of the segmentation.
    """
    return fv.accuracy(yp, y, axis=1)