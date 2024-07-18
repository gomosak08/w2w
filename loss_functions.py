import torch
import torch.nn as nn
import torch.nn.functional as F
from image_processing import MSTensorImage
from fastai.vision.core import TensorMask


class CombinedLoss(nn.Module):
    """
    A custom loss function combining Cross-Entropy Loss, Focal Loss, and Dice Loss.

    Attributes:
    - `weight` (Tensor, optional): A manual rescaling weight given to each class. Default is `None`.
    - `gamma` (float, optional): Focusing parameter for Focal Loss. Default is `2`.
    - `alpha` (float, optional): Weight for Cross-Entropy Loss. Default is `1`.
    - `beta` (float, optional): Weight for Focal Loss. Default is `1`.
    - `delta` (float, optional): Weight for Dice Loss. Default is `1`.
    """

    def __init__(self, weight=None, gamma=2, alpha=1, beta=1, delta=1):
        """
        Initializes the CombinedLoss class.

        Parameters:
        - `weight` (Tensor, optional): A manual rescaling weight given to each class. Default is `None`.
        - `gamma` (float, optional): Focusing parameter for Focal Loss. Default is `2`.
        - `alpha` (float, optional): Weight for Cross-Entropy Loss. Default is `1`.
        - `beta` (float, optional): Weight for Focal Loss. Default is `1`.
        - `delta` (float, optional): Weight for Dice Loss. Default is `1`.
        """
        super(CombinedLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.delta = delta

    def forward(self, inputs, targets):
        """
        Computes the combined loss.

        Parameters:
        - `inputs` (Tensor): Predicted tensor from the model.
        - `targets` (Tensor): Ground truth tensor.

        Returns:
        - `Tensor`: The computed combined loss.
        """
        device = inputs.device
        if isinstance(inputs, (MSTensorImage, TensorMask)):
            inputs = inputs.data
        
        weight = self.weight.to(device) if self.weight is not None else None

        # Cross-Entropy Loss with weights
        ce_loss = F.cross_entropy(inputs, targets, weight=weight)

        # Focal Loss
        pt = torch.exp(-F.cross_entropy(inputs, targets, reduction='none'))
        focal_loss = ((1 - pt) ** self.gamma * F.cross_entropy(inputs, targets, reduction='none')).mean()

        # Dice Loss
        smooth = 1e-5
        inputs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).permute(0, 3, 1, 2).float()
        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        dice_loss = 1 - (2 * intersection + smooth) / (union + smooth)
        dice_loss = dice_loss.mean()

        # Combined Loss
        loss = self.alpha * ce_loss + self.beta * focal_loss + self.delta * dice_loss
        return loss