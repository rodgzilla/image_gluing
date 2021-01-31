import pdb
import logging
from pathlib import Path
import PIL # type: ignore
import click
from typing import Tuple
from guppy import hpy # type: ignore

import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import torch
from sklearn.metrics import pairwise_distances # type: ignore
from tqdm import tqdm # type: ignore

from utils import slice_image, glue_images, load_target_img, load_cifar_imgs # type: ignore
from perceptual_repr import extract_features # type: ignore

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger('Image gluing genetic algorithm')

def generate_random_individuals(
    block_array_size: Tuple[int, int, int],
    img_database_size: int
) -> np.ndarray:
    return np.random.randint(
        low = 0,
        high = img_database_size,
        size = block_array_size
    )

def save_individuals(
    individuals: np.ndarray,
    scores: np.ndarray,
    img_database: np.ndarray,
    generation_idx: int,
    output_folder: Path
) -> None:
    imgs_to_glue = img_database[individuals]
    glued_imgs = glue_images(imgs_to_glue)
    for img_idx, (img, score) in enumerate(zip(glued_imgs, scores)):
        pil_img = PIL.Image.fromarray(img)
        img_fn = output_folder / f'gen_{generation_idx:04d}_img_{img_idx}_score_{score}.png'
        pil_img.save(img_fn)

def score_individuals(
    individuals: np.ndarray,
    img_database: np.ndarray,
    target_img_repr: np.ndarray,
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

def score_and_sort_population(
    population: np.ndarray,
    img_database: np.ndarray,
    target_img_repr: np.ndarray,
    batch_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    population_scores = score_individuals(
        individuals = population,
        img_database = img_database,
        target_img_repr = target_img_repr,
        batch_size = batch_size
    )
    sorted_indices = np.argsort(population_scores)
    population_scores = population_scores[sorted_indices]
    population = population[sorted_indices]

    return population, population_scores

def reproductions(
    population: np.ndarray,
    first_parent_indices: np.ndarray,
    second_parent_indices: np.ndarray
) -> np.ndarray:
    first_parents = population[first_parent_indices]
    second_parents = population[second_parent_indices]
    mask = np.random.random(first_parents.shape) > .5

    return mask * first_parents + (1 - mask) * second_parents

def mutation(
    population: np.ndarray,
    img_database_size: int,
    n_mutation: int,
    proba: float
) -> np.ndarray:
    individuals_to_mutate = population[
        np.random.permutation(len(population))[:n_mutation]
    ].copy()
    mutation_mask = np.random.random(individuals_to_mutate.shape) < proba
    img_to_mutate = individuals_to_mutate[mutation_mask]
    img_to_mutate = np.random.randint(
        low = 0,
        high = img_database_size,
        size = len(img_to_mutate)
    )

    return individuals_to_mutate

def remove_duplicate_individuals(population: np.ndarray) -> np.ndarray:
    population = np.unique(population, axis = 0)

    return population

def run_generation(
    pop_size: int,
    n_gen: int,
    n_mutation: int,
    mutation_proba: float,
    n_reprod: int,
    n_select: int,
    n_new_ind: int,
    block_size:int,
    target_img: np.ndarray,
    img_database: np.ndarray,
    batch_size: int,
    output_folder: Path,
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
    population, population_scores = score_and_sort_population(
        population = population,
        img_database = img_database,
        target_img_repr = target_img_repr,
        batch_size = batch_size
    )
    for generation_idx in tqdm(range(n_gen)):
        # h = hpy()
        # print(h.heap())
        logger.info(f'Start of generation {generation_idx}')
        logger.info(f'\nTop scores {population_scores[:5]}')
        logger.info(f'Performing reproductions')
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

        intermediate_population = np.concatenate((
            population,
            reprod_children
        ))
        logger.info(f'Performing mutations')
        mutated_population = mutation(
            population = intermediate_population,
            img_database_size = len(img_database),
            n_mutation = n_mutation,
            proba = mutation_proba
        )

        logger.info('Generating random individuals')
        new_individuals = generate_random_individuals(
            block_array_size = (
                n_new_ind,
                target_img_slices.shape[0],
                target_img_slices.shape[1]
            ),
            img_database_size = len(img_database)
        )

        logger.info(f'Scoring population')
        new_population = np.concatenate((
            intermediate_population,
            mutated_population,
            new_individuals
        ))
        new_population, new_population_scores = score_and_sort_population(
            population = new_population,
            img_database = img_database,
            target_img_repr = target_img_repr,
            batch_size = batch_size
        )
        population = new_population[:pop_size]
        population_scores = population_scores[:pop_size]

        # Duplication removal
        population = remove_duplicate_individuals(population)
        if len(population) < pop_size:
            logger.info(f'{pop_size - len(population)} duplicates removed, '
                        'filling back the population')
            new_individuals = generate_random_individuals(
                block_array_size = (
                    pop_size - len(population),
                    target_img_slices.shape[0],
                    target_img_slices.shape[1]
                ),
                img_database_size = len(img_database)
            )
            population = np.concatenate((
                population,
                new_individuals
            ))
            population, population_scores = score_and_sort_population(
                population = population,
                img_database = img_database,
                target_img_repr = target_img_repr,
                batch_size = batch_size
            )

        if generation_idx % 5 == 0:
            save_individuals(
                population[:3],
                population_scores[:3],
                img_database,
                generation_idx,
                output_folder
            )


@click.command()
@click.argument('target_img_fn', type = click.Path(exists = True))
@click.argument('output_folder', type = click.Path(exists = True))
@click.argument('target_img_height', type = int)
@click.argument('target_img_width', type = int)
@click.argument('pop_size', type = int)
@click.argument('n_gen', type = int)
@click.argument('n_reprod', type = int)
@click.argument('n_select', type = int)
@click.argument('n_mutation', type = int)
@click.argument('mutation_proba', type = float)
@click.argument('n_new_ind', type = int)
def main(
    target_img_fn: str,
    output_folder: str,
    target_img_height: int,
    target_img_width: int,
    pop_size: int,
    n_gen: int,
    n_reprod: int,
    n_select: int,
    n_mutation: int,
    mutation_proba: float,
    n_new_ind: int
) -> None:
    target_img_path = Path(target_img_fn)
    output_folder_path = Path(output_folder)
    logger.info(f'Target image path: {target_img_path}')
    logger.info(f'Target image height: {target_img_height}')
    logger.info(f'Target image width: {target_img_width}')
    logger.info(f'Output folder path: {output_folder_path}')
    logger.info(f'Population size: {pop_size}')
    logger.info(f'Number of generation: {n_gen}')
    logger.info(f'Number of reproductions: {n_reprod}')
    logger.info(f'First parent selection range: {n_select}')
    logger.info(f'Number of mutated individuals: {n_mutation}')
    logger.info(f'Mutation image change probability: {100 * mutation_proba:5.3f}%')
    logger.info(f'Number of new random individuals at each generation: {n_new_ind}')

    target_img = load_target_img(
        target_img_path,
        target_img_height,
        target_img_width
    )

    img_database = load_cifar_imgs('data')
    logger.info(f'Image database shape: {img_database.shape}')

    run_generation(
        pop_size = pop_size,
        n_mutation = n_mutation,
        mutation_proba = mutation_proba,
        n_gen = n_gen,
        n_reprod = n_reprod,
        n_select = n_select,
        n_new_ind = n_new_ind,
        block_size = 32,
        target_img = target_img,
        img_database = img_database,
        batch_size = 1,
        output_folder = output_folder_path
    )

if __name__ == '__main__':
    main()
