from image_setup import *
from image_generation import *
from image_processing import *

def generate_images(image_count, config_setup, config_process=None, config_characters=None, box=True):
    """
    Generate images with random moves and save them to the output folder.
    :param image_count: number of images to generate
    :param config_setup: dictionary with setup parameters
    :param config_process: dictionary with processing parameters
    :param config_characters: dictionary with character parameters
    :param box: boolean, if True, a box surrounds the texr
    """
    if not os.path.exists(config_setup['data_output_folder']):
        os.makedirs(config_setup['data_output_folder'])
    for i in range(image_count):
        move = get_random_move(config_setup)
        if config_process is not None:
            new_move = add_character_change(move, config_process, config_characters)
        else:
            new_move = move
        img, bgc = generate_image(new_move, config_setup, box)
        if config_process is not None:
            img = add_processing(img, config_process, bgc)
        img = img.convert('L')
        save_result(str(i), img, move, config_setup)


if __name__ == "__main__":
    generate_images(10000, config_setup, config_process, config_characters)