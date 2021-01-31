from pathlib import Path

import PIL # type: ignore
import numpy as np # type: ignore
import torchvision.datasets as datasets # type: ignore

def load_target_img(img_path: Path, img_height: int,
                    img_width: int
) -> np.ndarray:
    return np.array(PIL.Image.open(img_path).resize(
        (img_width, img_height)
    )) / 256

def load_cifar_imgs(data_folder: Path) -> np.ndarray:
    cifar_train = datasets.CIFAR10(
        root      = data_folder,
        train     = True,
        download  = True,
        transform = None
    )
    cifar_test = datasets.CIFAR10(
        root      = data_folder,
        train     = False,
        download  = True,
        transform = None
    )
    imgs = np.stack([
        img for img, _ in cifar_train + cifar_test
    ])

    return imgs

def slice_image(img: np.ndarray, block_size: int) -> np.ndarray:
    '''
    This function takes an image as input (a numpy array of shape
    [height, width, 3]) and slice it into blocks of
    `block_size x block_size` pixels. The output is of shape
    `[n_block_x, n_block_y, n_pixel_block_x, n_pixel_block_y, rgb]`
    '''
    grid_size = img.shape[0] // block_size
    grid      = img.reshape(
        grid_size,
        block_size,
        grid_size,
        block_size,
        3
    )
    grid      = grid.transpose((0, 2, 1, 3, 4))

    return grid

def glue_image(imgs_to_glue: np.ndarray) -> np.ndarray:
    '''
    This function is the inverse of slice image, it glues many images
    together to form a big one. The input if of shape
    `[n_block_x, n_block_y, n_pixel_block_x, n_pixel_block_y, rgb]`
    and the output is of shape
    `[n_block_x * n_pixel_block_x, n_block_y * n_pixel_block_y, rgb]
    '''
    glued_img = imgs_to_glue.transpose((0, 2, 1, 3, 4))
    glued_img = glued_img.reshape(
        glued_img.shape[0] * glued_img.shape[1],
        glued_img.shape[2] * glued_img.shape[3],
        3
    )

    return glued_img
