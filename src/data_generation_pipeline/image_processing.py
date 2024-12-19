import json, random, cv2
import numpy as np
from PIL import Image

config_process = json.load(open('config_process.json', 'r'))
config_characters = json.load(open('config_characters.json', 'r'))

def add_rotation(image, config, bg_color):
    """
    Add rotation to the image
    :param image: PIL image
    :param config: configuration dictionary
    :param bg_color: background color
    :return: PIL image
    """
    p = config['rotation_probability']
    if random.random() > p:
        return image
    angle_range = config['rotation_range']
    rotation = random.randint(angle_range[0], angle_range[1])
    return image.rotate(rotation, resample=Image.BICUBIC, fillcolor=bg_color)

def add_translation(image, config, bg_color):
    """
    Add translation to the image
    :param image: PIL image
    :param config: configuration dictionary
    :param bg_color: background color
    :return: PIL image
    """
    p = config['translation_probability']
    if random.random() > p:
        return image
    tr_range = config['translation_range']
    x_translation = random.randint(tr_range[0], tr_range[1])
    y_translation = random.randint(tr_range[0], tr_range[1])
    return image.transform(image.size, Image.AFFINE, (1, 0, x_translation, 0, 1, y_translation), fillcolor=bg_color)

def add_shear(image, config, bg_color):
    """
    Add shear to the image
    :param image: PIL image
    :param config: configuration dictionary
    :param bg_color: background color
    :return: PIL image
    """
    p = config['shear_probability']
    if random.random() > p:
        return image
    shear_range = config['shear_range']
    shear = random.uniform(shear_range[0], shear_range[1])
    return image.transform(image.size, Image.AFFINE, (1, shear, 0, 0, 1, 0), fillcolor=bg_color)

def add_perspective(image, config, bg_color):
    """
    Add perspective to the image, by changing the viewing angle
    :param image: PIL image
    :param config: configuration dictionary
    :param bg_color: background color
    :return: PIL image
    """
    p = config['perspective_probability']
    if random.random() > p:
        return image
    perspective_range = config['perspective_range']
    perspective = random.uniform(perspective_range[0], perspective_range[1])
    width, height = image.size
    reduction_pixels = int(height * perspective)
    src_points = [(0, 0), (width, 0), (0, height), (width, height)]
    dst_points = [(0, 0), (width, reduction_pixels), (0, height), (width, height - reduction_pixels)]

    matrix = []
    for p1, p2 in zip(dst_points, src_points):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])
    A = np.matrix(matrix, dtype=np.float64)
    B = np.array(src_points).reshape(8)
    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    coeffs = np.array(res).reshape(8)
    img = image.transform((width, height), Image.PERSPECTIVE, coeffs, Image.BICUBIC, fillcolor=bg_color)
    return img

def add_zoom(image, config, bg_color=None):
    """
    Add zoom to the image
    :param image: PIL image
    :param config: configuration dictionary
    :param bg_color: background color
    :return: PIL image
    """
    p = config['zoom_probability']
    if random.random() > p:
        return image
    zoom_range = config['zoom_range']
    zoom = random.uniform(zoom_range[0], zoom_range[1])
    width, height = image.size
    new_width = int(width * zoom)
    new_height = int(height * zoom)
    img = image.resize((new_width, new_height), resample=Image.BICUBIC)
    return img

def add_brightness(image, config, bg_color=None):
    """
    Add brightness to the image
    :param image: PIL image
    :param config: configuration dictionary
    :param bg_color: background color
    :return: PIL image
    """
    p = config['brightness_probability']
    if random.random() > p:
        return image
    brightness_range = config['brightness_range']
    brightness = random.uniform(brightness_range[0], brightness_range[1])
    img = np.array(image)
    img = img * brightness
    img = np.clip(img, 0, 255)
    img = Image.fromarray(img.astype(np.uint8))
    return img

def add_gaussian_noise(image, config, bg_color=None):
    """
    Add gaussian noise to the image
    :param image: PIL image
    :param config: configuration dictionary
    :param bg_color: background color
    :return: PIL image
    """
    p = config['gaussian_noise_probability']

    if random.random() > p:
        return image
    noise_range = config['gaussian_noise_range']
    noise = random.uniform(noise_range[0], noise_range[1])
    img = np.array(image)
    noise = np.random.normal(0, noise, img.shape)
    img = img + noise
    img = np.clip(img, 0, 255)
    img = Image.fromarray(img.astype(np.uint8))
    return img

def add_salt_pepper_noise(image, config, bg_color=None):
    """
    Add salt and pepper noise to the image
    :param image: PIL image
    :param config: configuration dictionary
    :param bg_color: background color
    :return: PIL image
    """
    p = config['salt_and_pepper_noise_probability']
    if random.random() > p:
        return image
    noise_range = config['salt_and_pepper_noise_range']
    noise = random.uniform(noise_range[0], noise_range[1])
    img = np.array(image)
    salt_vs_pepper = 0.5
    amount = noise
    num_salt = np.ceil(amount * img.size * salt_vs_pepper)
    num_pepper = np.ceil(amount * img.size * (1.0 - salt_vs_pepper))
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    img[coords[0], coords[1], :] = 255
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
    img[coords[0], coords[1], :] = 0
    img = Image.fromarray(img.astype(np.uint8))
    return img

def add_gaussian_blur(image, config, bg_color=None):
    """
    Add gaussian blur to the image
    :param image: PIL image
    :param config: configuration dictionary
    :param bg_color: background color
    :return: PIL image
    """
    p = config['gaussian_blur_probability']
    if random.random() > p:
        return image
    blur_range = config['gaussian_blur_range']
    blur = random.randint(blur_range[0], blur_range[1])
    blur = 2 * blur + 1
    img = np.array(image)
    img = cv2.GaussianBlur(img, (blur, blur), 0)
    img = Image.fromarray(img)
    return img

def add_character_change(text, config, config_characters):
    """
    Change characters in the text, by substituting character with similar ones according to the configuration 
    :param text: string
    :param config: configuration dictionary
    :param config_characters: configuration dictionary
    :return: string
    """
    p = config['character_change_probability']
    if random.random() > p:
        return text
    new_text = ''
    for char in text:
        for key, value in config_characters.items():
            if char in key:
                new_text += np.random.choice(value)
                break
        else:
            new_text += char
    return new_text

def add_processing(image, config, bg_color):
    """
    Add processing to the image
    :param image: PIL image
    :param config: configuration dictionary
    :param bg_color: background color
    :return: PIL image
    """
    image = add_rotation(image, config, bg_color)
    image = add_translation(image, config, bg_color)
    image = add_shear(image, config, bg_color)
    image = add_perspective(image, config, bg_color)
    image = add_zoom(image, config, bg_color)
    image = add_brightness(image, config, bg_color)
    image = add_gaussian_noise(image, config, bg_color)
    image = add_salt_pepper_noise(image, config, bg_color)
    image = add_gaussian_blur(image, config, bg_color)
    return image