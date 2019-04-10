import pdb
import numpy as np
import matplotlib.pyplot as plt
import PIL

import torch
import torchvision.models.vgg as models
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader

normalize = transforms.Normalize(
    mean = [0.485, 0.456, 0.406],
    std  = [0.229, 0.224, 0.225]
)
# This way of dealing with devices is not flexible, we should
# change it.
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg16     = models.vgg16(pretrained = True).to(device)

def run_model_inference(model, image_dataset, batch_size):
    image_loader = DataLoader(
        dataset    = image_dataset,
        batch_size = batch_size,
        shuffle    = False,
    )

    result = []
    for img_batch, in image_loader:
        img_batch = img_batch.to(device)
        vgg_repr  = model(img_batch)
        vgg_repr  = vgg_repr.detach().cpu()
        result.append(vgg_repr)
    result = torch.cat(result, dim = 0)
    result = result.view(len(result), -1)

    return result

def extract_features(images, layer_idx, batch_size):
    images        = images.transpose((0, 3, 1, 2))
    images_tensor = torch.tensor(images, dtype = torch.float32)
    # This is obviously wrong but I haven't found how to do it
    # properly
    for i, img in enumerate(images_tensor):
        images_tensor[i] = normalize(img)

    image_dataset = TensorDataset(images_tensor)
    model         = vgg16.features[:layer_idx]
    reprs         = run_model_inference(model, image_dataset, 1)
    reprs         = reprs.numpy()

    return reprs

if __name__ == '__main__':
    img_1 = np.array(
        PIL.Image.open(
            '../../data/raw/cat.jpg'
        ).resize((256, 256)).rotate(-90)
    )
    img_2 = np.array(
        PIL.Image.open(
            '../../data/raw/cat_snow.jpg'
        ).resize((256, 256))
    )
    images = np.stack((img_1, img_2))
    images = images / 255.
    extract_features(images, 7, 1)
