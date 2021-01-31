import pdb
from pathlib import Path
import PIL # type: ignore
import click
from typing import Tuple

import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import torch
from sklearn.metrics import pairwise_distances # type: ignore
from tqdm import tqdm

from utils import slice_image, glue_images, load_target_img, load_cifar_imgs # type: ignore
from perceptual_repr import extract_features # type: ignore

def generate_random_individuals(
    block_array_size: Tuple[int, int, int],
    img_database_size: int
) -> np.ndarray:
    return np.random.randint(
        low = 0,
        high = img_database_size,
        size = block_array_size
    )

def score_individuals(
    individuals: np.ndarray,
    img_database: np.ndarray,
    target_img_repr: torch.Tensor,
    batch_size: int
) -> np.ndarray:
    imgs_to_glue = img_database[individuals]
    glued_imgs = glue_images(imgs_to_glue)
    glued_imgs_repr = extract_features(
        images = glued_imgs,
        layer_idx = 7,
        batch_size = batch_size
    )
    scores = pairwise_distances(target_img_repr, glued_imgs_repr)

    return scores[0]

def reproductions(
        population: np.ndarray,
        first_parent_indices: np.ndarray,
        second_parent_indices: np.ndarray
) -> np.ndarray:
    first_parents = population[first_parent_indices]
    second_parents = population[second_parent_indices]
    mask = np.random.random(first_parents.shape) > .5

    return mask * first_parents + (1 - mask) * second_parents

def run_generation(
    pop_size: int,
    n_gen: int,
    n_mutation: int,
    n_reprod: int,
    n_select: int,
    block_size:int,
    target_img: np.ndarray,
    img_database: np.ndarray,
    batch_size: int
) -> None:
    target_img_slices = slice_image(
        img = target_img,
        block_size = block_size
    )
    target_img_repr = extract_features(
        images = target_img[None, ...],
        layer_idx = 7,
        batch_size = 1
    )
    population = generate_random_individuals(
        block_array_size = (
            pop_size,
            target_img_slices.shape[0],
            target_img_slices.shape[1]
        ),
        img_database_size = len(img_database)
    )
    population_scores = score_individuals(
        individuals = population,
        img_database = img_database,
        target_img_repr = target_img_repr,
        batch_size = batch_size
    )
    sorted_indices = np.argsort(population_scores)
    population_scores = population_scores[sorted_indices]
    population = population[sorted_indices]

    for generation_id in tqdm(range(n_gen)):
        reprod_parent_1 = np.random.randint(
            low = 0,
            high = n_select,
            size = (n_reprod, )
        )
        reprod_parent_2 = np.random.randint(
            low = 0,
            high = len(population),
            size = (n_reprod, )
        )
        reprod_children = reproductions(
            population,
            reprod_parent_1,
            reprod_parent_2
        )


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
    img_database = load_cifar_imgs('data')
    run_generation(
        pop_size = 30,
        n_mutation = 10,
        n_gen = 3,
        n_reprod = 10,
        n_select = 5,
        block_size = 32,
        target_img = target_img,
        img_database = img_database,
        batch_size = 1
    )
    # big_img_slices = slice_image(target_img, block_size = 32)
    # small_imgs = load_cifar_imgs('data')
    # target_img_repr = extract_features(
    #     images = target_img[None, ...],
    #     layer_idx = 7,
    #     batch_size = 1
    # )
    # individuals = generate_random_individuals(
    #     block_array_size = (
    #         5,
    #         big_img_slices.shape[0],
    #         big_img_slices.shape[1],
    #     ),
    #     img_database_size = len(small_imgs)
    # )
    # scores = score_individuals(
    #     individuals = individuals,
    #     img_database = small_imgs,
    #     target_img_repr = target_img_repr
    # )

    # print(scores)

if __name__ == '__main__':
    main()
