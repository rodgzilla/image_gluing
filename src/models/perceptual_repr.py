import pdb
from pathlib import Path
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import PIL # type: ignore

from sklearn.metrics import pairwise_distances # type: ignore

import torch # type: ignore
import torch.nn as nn # type: ignore
import torchvision.models.vgg as models # type: ignore
import torchvision.transforms as transforms # type: ignore
from torch.utils.data import TensorDataset, DataLoader, Dataset # type: ignore

normalize = transforms.Normalize(
    mean = [0.485, 0.456, 0.406],
    std  = [0.229, 0.224, 0.225]
)
# This way of dealing with devices is not flexible, we should
# change it.
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg16     = models.vgg16(pretrained = True).to(device)

def load_images(root: Path) -> np.ndarray:
    img_arrays = []
    for fn in root.iterdir():
        img_arrays.append(
            PIL.Image.open(
                fn
            ).resize((256, 256))
        )

    return np.stack(img_arrays) / 255

def run_model_inference(
    model: nn.Module,
    image_dataset: Dataset,
    batch_size: int
) -> torch.Tensor:
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
    result_tensor = torch.cat(result, dim = 0)
    result_tensor = result_tensor.view(len(result_tensor), -1)

    return result_tensor

def extract_features(images: np.ndarray, layer_idx: int,
                     batch_size: int
) -> np.ndarray:
    images        = images.transpose((0, 3, 1, 2))
    images_tensor = torch.tensor(images, dtype = torch.float32)
    # This is obviously wrong but I haven't found how to do it
    # properly
    for i, img in enumerate(images_tensor):
        images_tensor[i] = normalize(img)

    image_dataset = TensorDataset(images_tensor)
    model         = vgg16.features[:layer_idx]
    reprs         = run_model_inference(model, image_dataset, batch_size)
    reprs         = reprs.numpy()

    return reprs

if __name__ == '__main__':
    root = Path('data/raw')
    imgs = load_images(root)
    result = extract_features(imgs, 7, 2)
    distances = pairwise_distances(result)
    print(distances)
    pdb.set_trace()
    print(5)
