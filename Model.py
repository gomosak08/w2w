## #!/home/gomosak/cnf/bin/python
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

print(torch.cuda.is_available())
# Read the path from the text file
with open('routes.txt', 'r') as file:
    path_str = file.read().strip()  # Read the path and strip any surrounding whitespace
#routes.txt
# Convert the path string to a Path object
path = Path(path_str)

imgs_path = path/'data'
lbls_path = path/'masks'


print(f'Checking number of files - images:{len([f for f in imgs_path.iterdir()])}\
      masks:{len([f for f in lbls_path.iterdir()])}')


def order(x):
    try:
        return int(x.stem)
    except:
        return x.split('_')[-1][:-4]
    
img_path_1 = sorted([f for f in imgs_path.iterdir()], key = lambda x: int(x.stem))
msk_path_1 = sorted([f for f in lbls_path.iterdir()], key = lambda x: int(x.stem))
img_path = img_path_1[12]
msk_path = msk_path_1[12]
img = np.load(str(img_path))
msk = np.load(str(msk_path))

print(f'Checking shapes - image: {img.shape} mask: {msk.shape}')


# Example usage
img_size = (544, 480)
batch_size = 4

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

# Train the model
learner.fine_tune(300)

# Show results
show_segmentation_results(learner)


#print(learner.validate())
# Find optimal learning rate
lr_min = learner.lr_find(start_lr=1e-07, end_lr=10)

# Plot the learning rate finder results
#learner.recorder.plot_lr_find()
#plt.show()
#print(lr_min)

show_segmentation_results(learner)
# Save the model after training
# Save the model after training

models_numer = [d for d in os.listdir('models')]
numbers = [int(re.search(r'[0-9]+', run).group()) for run in models_numer if re.search(r'[0-9]+', run)]
highest_number = max(numbers) if numbers else 1

learner.save(f'model_{highest_number+1}')
print("model saved in",f' models/model_{highest_number+1}')
