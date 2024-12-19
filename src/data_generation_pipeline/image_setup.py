import random, os, json

config_setup = json.load(open('config_setup.json', 'r'))

data_output_folder = config_setup['data_output_folder']

# Create the output folder if it does not exist
if not os.path.exists(data_output_folder):
    os.makedirs(data_output_folder)


def get_moves(config):
    """
    Get the moves and their distribution from the moves_distribution_file
    :param config: the configuration dictionary
    :return: a tuple containing the moves and their distribution
    """
    moves_file = config['moves_distribution_file']
    moves = {}
    with open(moves_file, 'r') as f:
        moves = {a:int(b) for a,b in [line.strip().split(",") for line in f.readlines()]}
    keys, values = zip(*moves.items())
    return keys, values

def get_random_move(config):
    """
    Get a random move based on the distribution in the moves_distribution_file
    :param config: the configuration dictionary 
    :return: a random move
    """
    keys, values = get_moves(config)
    return random.choices(keys, weights=values)[0]

def get_font_paths(config):
    """
    Get the paths of all the fonts in the fonts_folder
    :param config: the configuration dictionary
    :return: a list containing the paths of all the fonts
    """
    fonts_folder = config['fonts_folder']
    font_paths = []
    for root, dirs, files in os.walk(fonts_folder):
        for file in files:
            if file.upper().endswith(".TTF") and not "BOLD" in file.upper():
                font_paths.append(os.path.join(root, file))
    return font_paths

def save_result(name, image, text, config):
    """
    Save the image and the text in the output folder
    :param name: the name of the file
    :param image: the image to save
    :param text: the text to save
    :param config: the configuration dictionary
    """
    folder_path = config['data_output_folder']
    image.save(os.path.join(folder_path,  name + ".png"))
    with open(os.path.join(folder_path, name + ".txt"), 'w') as f:
        f.write(text)