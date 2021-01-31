import pdb
from pathlib import Path
import PIL # type: ignore
import click
from typing import Tuple

import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import torch

from utils import slice_image, glue_image, load_target_img, load_cifar_imgs # type: ignore

def generate_random_individual(
    block_array_size: Tuple[int, int],
    img_database_size: int
) -> torch.Tensor:
    return torch.randint(
        low = 0,
        high = img_database_size,
        size = block_array_size
    )

def score_individual(
    individual: torch.Tensor,
    img_database: torch.Tensor,
    target = torch.Tensor
) -> float:
    imgs_to_glue = img_database[individual]
    glued_imgs = glue_image(imgs_to_glue)


@click.command()
@click.argument('target_img_fn', type = click.Path(exists = True))
@click.argument('target_img_height', type = int)
@click.argument('target_img_width', type = int)
def main(target_img_fn: str, target_img_height: int,
         target_img_width: int
) -> None:
    target_img_path = Path(target_img_fn)
    target_img = load_target_img(
        target_img_path,
        target_img_height,
        target_img_width
    )
    big_img_slices = slice_image(target_img, block_size = 32)
    small_imgs = load_cifar_imgs('data')
    individual = generate_random_individual(
        block_array_size = big_img_slices.shape[:2],
        img_database_size = len(small_imgs)
    )
    imgs_to_glue = small_imgs[individual]
    result_img = glue_image(imgs_to_glue)
    plt.imshow(result_img)
    plt.show()

if __name__ == '__main__':
    main()
