from PIL import ImageFont, ImageDraw, Image
import numpy as np
from image_setup import *


def dashed_line(draw, x1, y1, x2, y2, dash_length, dash_separation, fill="black", width=2):
    """
    Draw a dashed line between two points.
    :param draw: ImageDraw object
    :param x1: x-coordinate of the starting point
    :param y1: y-coordinate of the starting point
    :param x2: x-coordinate of the ending point
    :param y2: y-coordinate of the ending point
    :param dash_length: length of the dashes
    :param dash_separation: separation between dashes
    :param fill: line color
    :param width: line width
    """
    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    dash_count = int(length / (dash_length + dash_separation))

    # Calculate the gap between dashes
    dash_x = (x2 - x1) / dash_count
    dash_y = (y2 - y1) / dash_count

    for i in range(dash_count):
        # Calculate dash start and end points
        x_start = x1 + i * dash_x
        y_start = y1 + i * dash_y
        x_end = x_start + dash_length * (x2 - x1) / length
        y_end = y_start + dash_length * (y2 - y1) / length

        # Draw the dash
        draw.line([(x_start, y_start), (x_end, y_end)], fill=fill, width=width)

def draw_line(draw, x1, y1, x2, y2,style="solid",fill="black",width=2):
    """
    Draw a line between two points.
    :param draw: ImageDraw object
    :param x1: x-coordinate of the starting point
    :param y1: y-coordinate of the starting point
    :param x2: x-coordinate of the ending point
    :param y2: y-coordinate of the ending point
    :param style: line style, "solid" or "dashed"
    :param fill: line color
    :param width: line width
    """
    if style=="solid":
        draw.line((x1, y1, x2, y2), fill=fill, width=width)
    elif style=="dashed":
        dashed_line(draw, x1, y1, x2, y2, dash_length=5, dash_separation=4, fill=fill, width=width)
    else:
        print("Unknown style")
        return

def select_random_font(config):
    """
    Select a random font and text color from the configuration.
    :param config: dictionary with configuration parameters
    :return: font and text color
    """
    text_colors = config['text_colors']
    font_paths = get_font_paths(config)
    font_path = random.choice(font_paths)
    font_size = random.randint(30, 51)
    font = ImageFont.truetype(font_path, size=font_size)
    text_color = random.choice(text_colors)
    return font, text_color

def select_line_properties(config, background_color):
    """
    Select line properties from the configuration.
    :param config: dictionary with configuration parameters
    :param background_color: background color
    :return: line style, line color, vertical line color, horizontal line width, vertical line width
    """
    line_styles = config['line_styles']
    line_colors = config['line_colors']
    line_style = random.choice(line_styles)
    line_color = random.choice(line_colors)
    vertical_line_color = random.choice([line_color, background_color])
    line_width_horizontal = int(random.gauss(2, 1))
    line_width_vertical = int(random.gauss(2, 1)) if vertical_line_color == line_color else int(random.gauss(8, 1))
    return line_style, line_color, vertical_line_color, line_width_horizontal, line_width_vertical

def initialize_canvas(config):
    """
    Initialize the canvas for the image generation.
    :param config: dictionary with configuration parameters
    :return: image, rectangle properties, draw object
    """
    background_colors = config['background_colors']
    background_color = random.choice(background_colors)
    image_size = (int(random.gauss(180, 5)), int(random.gauss(80, 5)))
    canvas_size = (300, 200)
    box_shift_x = (canvas_size[0] - image_size[0]) / 2
    box_shift_y = (canvas_size[1] - image_size[1]) / 2
    img = Image.new("RGB", canvas_size, background_color)

    rectangle_width = random.gauss(130, 5)
    rectangle_height = random.gauss(52, 3)
    draw = ImageDraw.Draw(img)
    draw_point = (random.gauss(box_shift_x + (image_size[0] - rectangle_width) / 2, 3),
                  random.gauss(box_shift_y + (image_size[1] - rectangle_height) / 2, 3))
    bottom_right = (draw_point[0] + rectangle_width, draw_point[1] + rectangle_height)

    rectangle_properties = {
        "draw_point": draw_point,
        "bottom_right": bottom_right,
        "rectangle_width": rectangle_width,
        "rectangle_height": rectangle_height,
        "box_shift_x": box_shift_x,
        "box_shift_y": box_shift_y,
        "canvas_size": canvas_size,
        "image_size": image_size,
        "background_color": background_color
    }

    return img, rectangle_properties, draw

def draw_box(draw, rectangle_properties, config):
    """
    Draw a box around the text.
    :param draw: ImageDraw object
    :param rectangle_properties: dictionary with rectangle properties
    :param config: dictionary with configuration
    """
    background_color = rectangle_properties["background_color"]
    line_style, line_color, vertical_line_color, line_width_horizontal, line_width_vertical = select_line_properties(config, background_color)
    dp = rectangle_properties["draw_point"]
    br = rectangle_properties["bottom_right"]
    has_cell_top = random.random() < 0.8
    has_cell_bottom = random.random() < 0.8
    has_cell_left = random.random() < 0.8
    has_cell_right = random.random() < 0.8

    draw_horizontal_left = 0 if has_cell_left else dp[0]
    draw_horizontal_right = rectangle_properties["canvas_size"][0] if has_cell_right else br[0]
    draw_vertical_top = 0 if has_cell_top else dp[1]
    draw_vertical_bottom = rectangle_properties["canvas_size"][1] if has_cell_bottom else br[1]

    draw_line(draw, draw_horizontal_left, dp[1], draw_horizontal_right, dp[1], style=line_style, fill=line_color, width=line_width_horizontal)
    draw_line(draw, draw_horizontal_left, br[1], draw_horizontal_right, br[1], style=line_style, fill=line_color, width=line_width_horizontal)
    
    if random.random() < 0.8:
        draw_line(draw, dp[0], draw_vertical_top, dp[0], draw_vertical_bottom, style=line_style, fill=vertical_line_color, width=line_width_vertical)
    if random.random() < 0.8:
        draw_line(draw, br[0], draw_vertical_top, br[0], draw_vertical_bottom, style=line_style, fill=vertical_line_color, width=line_width_vertical)

def draw_text(draw, rectangle_properties, text, font, text_color):
    """
    Draw text inside the box.
    :param draw: ImageDraw object
    :param rectangle_properties: dictionary with rectangle properties
    :param text: text to draw
    :param font: font object
    :param text_color: text color
    """
    dp = rectangle_properties["draw_point"]
    rw = rectangle_properties["rectangle_width"]
    rh = rectangle_properties["rectangle_height"]
    shift_x = random.gauss(0, 0.05) * rw
    shift_y = random.gauss(0, 0.05) * rh
    text_x = rw / 2 + dp[0] + shift_x
    text_y = rh * 0.9 + dp[1] + shift_y
    text_point = (text_x, text_y)
    draw.text(text_point, text, font=font, anchor="ms", fill=text_color)

def add_surrounding_text(draw, rectangle_properties, config, font, text_color):
    """
    Add text outside the box, based on another random move with the same configuration.
    :param draw: ImageDraw object
    :param rectangle_properties: dictionary with rectangle properties
    :param config: dictionary with configuration
    :param font: font object
    :param text_color: text color
    """
    dp = rectangle_properties["draw_point"]
    rw = rectangle_properties["rectangle_width"]
    rh = rectangle_properties["rectangle_height"]
    shift_x = random.gauss(0, 0.05) * rw
    shift_y = random.gauss(0, 0.05) * rh
    text_x = rw / 2 + dp[0] + shift_x
    text_y = rh * 0.9 + dp[1] + shift_y
    if random.random() < 0.8:
        draw.text((text_x, text_y - rh), get_random_move(config), font=font, anchor="ms", fill=text_color)

def generate_image(text, config, box=True):
    """
    Generate an image with text and optional box.
    :param text: text to draw
    :param config: dictionary with configuration parameters
    :param box: boolean, if True, a box surrounds the text
    :return: image, background color
    """
    img, rectangle_properties, draw = initialize_canvas(config)
    font, text_color = select_random_font(config)
    if box: draw_box(draw, rectangle_properties, config)
    draw_text(draw, rectangle_properties, text, font, text_color)
    if box: add_surrounding_text(draw, rectangle_properties, config, font, text_color)
    cs = rectangle_properties["canvas_size"]
    image_size = rectangle_properties["image_size"]
    img = img.crop(((cs[0] - image_size[0]) / 2, (cs[1] - image_size[1]) / 2, (cs[0] + image_size[0]) / 2, (cs[1] + image_size[1]) / 2))
    return img, rectangle_properties["background_color"]