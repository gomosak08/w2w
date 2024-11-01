from fastai.vision.all import * 

import matplotlib.pyplot as plt
import numpy as np
from path import Path 
import torch 
from torchvision import transforms as T 

from data_loading import load_data
from metrics import seg_accuracy
from loss_functions import CombinedLoss
from model_creation import create_model, create_learner
from visualization import show_segmentation_results
from image_processing import MSTensorImage

with open('routes.txt', 'r') as file:
    path_str = file.read().strip()  # Read the path and strip any surrounding whitespace
#routes.txt
# Convert the path string to a Path object
path = Path(path_str)

imgs_path = path/'data'
lbls_path = path/'masks'

# Example usage
img_size = (544, 480)
batch_size = 4

img_path_1 = sorted([f for f in imgs_path.iterdir()], key = lambda x: int(x.stem))
# Load data
dls = load_data(imgs_path, lbls_path, img_size, batch_size)

# Create model
model = create_model("resnet101", in_channels=11, classes=7)

# Create loss function
loss_func = CombinedLoss()
opt_func = SGD
wd = 0.00039897560969184224



# Create learner
learner = create_learner(model, loss_func, opt_func=partial(opt_func, wd=wd), db=dls, metrics=[seg_accuracy])
learner.model.eval()
img_path = img_path_1[12]
# Make an inference
#pred_class, pred_idx, outputs = learner.predict(img_path)
print(lbls_path,img_path)

name = lbls_path
im = torch.from_numpy(np.load(img_path))#.type(torch.float32)

img = MSTensorImage(im)
pred = learner.predict(img)
#print(pred[2])
#print(pred[0].argmax(dim=0))
#pred_mask = pred.argmax(dim=0).to('cpu')  # Get the channel with the highest probability
img = pred[0].argmax(dim=0)
# Plot the predicted mask
plt.imshow(img, cmap='inferno', vmax=7, vmin=0)
plt.title("Predicted Mask")
plt.colorbar()  # Optional: adds a color bar to indicate scale
plt.axis('off')  # Hide the axes if desired
plt.show()