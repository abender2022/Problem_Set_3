# Flowers 102

## Data Loading

In this section, we will go through the process of loading the Flowers 102 dataset, preparing the data for analysis, and visualizing a sample image from the dataset.

### Loading Libraries

We begin by importing the necessary Python libraries, including `matplotlib.pyplot` for image visualization. We also define a custom function called `plot` that allows us to display images with optional titles.

```python
import matplotlib.pyplot as plt

# Define a function for plotting images
def plot(x, title=None):
    # Move tensor to CPU and convert to numpy
    x_np = x.cpu().numpy()

    # If tensor is in (C, H, W) format, transpose to (H, W, C)
    if x_np.shape[0] == 3 or x_np.shape[0] == 1:
        x_np = x_np.transpose(1, 2, 0)

    # If grayscale, squeeze the color channel
    if x_np.shape[2] == 1:
        x_np = x_np.squeeze(2)

    x_np = x_np.clip(0, 1)

    # Create a matplotlib figure and axis for plotting
    fig, ax = plt.subplots()
    if len(x_np.shape) == 2:  # Grayscale
        im = ax.imshow(x_np, cmap='gray')
    else:
        im = ax.imshow(x_np)
    plt.title(title)
    ax.axis('off')
    fig.set_size_inches(10, 10)
    plt.show()
```

### Downloading and Extracting the Dataset

Next, we download and extract the Flowers 102 dataset. The dataset consists of images of various flowers, and we'll use it for further analysis. Here are the steps involved:

1. Download the dataset labels and image zip file.
2. Unzip the image files.

```python
# Downloading and extracting the dataset
# Uncomment the following lines if you are running this in a Jupyter Notebook
!wget https://gist.githubusercontent.com/JosephKJ/94c7728ed1a8e0cd87fe6a029769cde1/raw/403325f5110cb0f3099734c5edb9f457539c77e9/Oxford-102_Flower_dataset_labels.txt
!wget https://s3.amazonaws.com/content.udacity-data.com/courses/nd188/flower_data.zip
!unzip 'flower_data.zip'
```

### Loading and Preprocessing the Data

Now, we load and preprocess the dataset using PyTorch's `datasets` and `transforms` modules. We also load the dataset labels into a Pandas DataFrame for reference.

```python
import torch
from torchvision import datasets, transforms
import os
import pandas as pd

# Directory and transforms
data_dir = '/content/flower_data/'
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Define data transformations
data_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Load the dataset using ImageFolder
dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transform)
dataset_labels = pd.read_csv('Oxford-102_Flower_dataset_labels.txt', header=None)[0].str.replace("'", "").str.strip()

# Load the dataset into a DataLoader for batching
dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)

# Extract the batch of images and labels
images, labels = next(iter(dataloader))

print(f"Images tensor shape: {images.shape}")
print(f"Labels tensor shape: {labels.shape}")
```

Finally, we visualize the 11th image from the dataset along with its corresponding label:

```python
i = 11
# Plot the 11th image from the dataset with its label
plot(images[i], dataset_labels[i])
```

This concludes the data loading and preprocessing section, and we move on to using a pretrained AlexNet for image classification.

## Pretrained AlexNet

In this section, we'll use a pretrained AlexNet model to classify an image from the Flowers 102 dataset.

### Loading Pretrained AlexNet

We start by loading a pretrained AlexNet model and some associated labels for classifying the images. We also define a preprocessing transformation for the input image.

```python
import torch
from torchvision import models, transforms
import requests
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define AlexNet model
alexnet = models.alexnet(pretrained=True).to(device)
labels = {int(key): value for (key, value) in requests.get('https://s3.amazonaws.com/mlpipes/pytorch-quick-start/labels.json').json().items()}

# Transform image for use in the model
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and preprocess the image
img = images[i]
from torchvision.transforms import ToPILImage
to_pil = ToPILImage()
img = to_pil(img)
img_t = preprocess(img).unsqueeze_(0).to(device)

# Print the shape of the preprocessed image tensor
img_t.shape

# Classify the image with AlexNet
scores, class_idx = alexnet(img_t).max(1)
print('Predicted class:', labels[class_idx.item()])
```

In this code, we load a pretrained AlexNet model and define a transformation to preprocess the input image. We then load the 11th image from our dataset, preprocess it, and use the AlexNet model to classify it, printing out the predicted class label.

## Finetuning

In this section, we explore the weights of various layers in the AlexNet model and visualize feature maps with filters.

### Extracting Layer Weights

We begin by extracting the weights of several layers in the AlexNet model, including convolutional layers and classifier layers.

```python
# Extract the weights of various layers in AlexNet
w0 = alexnet.features[0].weight.data
w1 = alexnet.features[3].weight.data
w2 = alexnet.features[6].weight.data
w3 = alexnet.features[8].weight.data
w4 = alexnet.features[10].weight.data
w5 = alexnet.classifier[1].weight.data
w6 = alexnet.classifier[4].weight.data
w7 = alexnet.classifier[6].weight.data
```

### Scaling and Visualizing Images

We define functions to scale and visualize image tensors:

```python
# Define a function to scale an image tensor
def scale(img):
    # Normalize the NumPy array to the range [0, 1]
    max_value = img.max()
    min_value = img.min()
    normalized_array = (img - min_value) / (max_value - min_value)
    return normalized_array

# Define a function to plot an image tensor
def tensor_plot(img_t, index=0):
   
