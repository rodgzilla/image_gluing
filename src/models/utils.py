def slice_image(img, block_size):
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

def glue_image(imgs_to_glue):
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
