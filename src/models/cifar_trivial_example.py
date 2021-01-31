import pdb
import sys
import PIL # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from collections import defaultdict

from sklearn.metrics import pairwise_distances # type: ignore

import torchvision.datasets as datasets # type: ignore

from utils import slice_image, glue_images # type: ignore

def load_CIFAR10(data_folder = '../../data'):
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
    cat_decoder = {
        0 : 'airplane',
        1 : 'automobile',
        2 : 'bird',
        3 : 'cat',
        4 : 'deer',
        5 : 'dog',
        6 : 'frog',
        7 : 'horse',
        8 : 'ship',
        9 : 'truck'
    }

    imgs_by_cat = defaultdict(list)
    for img, cat in cifar_train + cifar_test:
        imgs_by_cat[cat_decoder[cat]].append(img)

    return imgs_by_cat

def load_target_image(target_filename):
    # target_img = PIL.Image.open('../data/raw/cat_snow.jpg')
    target_img = PIL.Image.open(target_filename)
    targ_img   = np.array(target_img)

    return targ_img

def compute_mean_rgb_CIFAR(imgs_by_cat, block_size, cat = None):
    if cat:
        cifar_imgs = imgs_by_cat[cat]
    else:
        cifar_imgs = [img for cat_imgs in imgs_by_cat.values() for img in cat_imgs]
    cifar_imgs     = [img.resize((block_size, block_size)) for img in cifar_imgs]
    cifar_imgs     = np.stack([np.array(img) for img in cifar_imgs])
    cifar_mean_rgb = cifar_imgs.reshape(
        len(cifar_imgs),
        -1,
        3
    ).mean(axis = -2)

    return cifar_imgs, cifar_mean_rgb

def compute_pairwise_block_database_img_dist(big_img_block_rgb, cifar_mean_rgb):
    pairwise_euc_dist = pairwise_distances(
        big_img_block_rgb.reshape(-1, 3), # we have to reshape the target image
                                          #blocks array into a [x, 3] array
        cifar_mean_rgb,
        metric = 'euclidean'
    )
    pairwise_euc_dist = pairwise_euc_dist.reshape(*big_img_block_rgb.shape[:2], -1)

    return pairwise_euc_dist

def main(target_filename, block_size = 32):
    pdb.set_trace()
    imgs_by_cat       = load_CIFAR10('data')
    targ_img          = load_target_image(target_filename)
    big_img_slices    = slice_image(targ_img, block_size)
    n_block_x         = big_img_slices.shape[0]
    n_block_y         = big_img_slices.shape[1]
    big_img_block_rgb = big_img_slices.reshape(
        n_block_x,
        n_block_y,
        -1,
        3
    ).mean(axis = -2)
    cifar_imgs, cifar_mean_rgb = compute_mean_rgb_CIFAR(imgs_by_cat, block_size)
    pairwise_euc_dist          = compute_pairwise_block_database_img_dist(
        big_img_block_rgb,
        cifar_mean_rgb
    )
    best_block_replacement     = pairwise_euc_dist.argmin(axis = -1)
    imgs_to_glue               = cifar_imgs[best_block_replacement]
    glued_img                  = glue_images(imgs_to_glue[None, ...])
    plt.figure(figsize = (15, 15))
    plt.imshow(np.squeeze(glued_img))
    plt.show()

if __name__ == '__main__':
    target_filename = sys.argv[1]
    main(target_filename, 16)
