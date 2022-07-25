"""
Script for recreating image using genetic alghoritm.
"""

import string
import random
from skimage.metrics import peak_signal_noise_ratio as psns # For image similarity evaluation
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import imageio # For gif saving

#Load and show original image for tracking (convert to black and white)
original_image = Image.open("image.jpg").convert("L")
original_height, original_width  = original_image.size
cv2.imshow("Original", np.array(original_image))
cv2.imwrite("original.jpg",np.array(original_image))

#Adjust hyperparameters
NUMBER_OF_GENERATIONS = 7500
POPULATION_NUMBER = 50  # How many images in 1 generation (without elitism)
MUTATION_CHANCE = 0.1  # Chance of mutating (adding random shapes)
MUTATION_STRENGTH = 1  # How many shapes to add in mutation
ELITISM = True  # Turn on/off elitism (transfering best images to next generation without crossover)
ELITISM_NUMBER = 4  # How many best images transfer to next generation (elitism)
STARTING_SHAPE_NUMBER = 6  # How many shapes to draw on each image in first generation

PRINT_EVERY_GEN = 25  # Print fitness value every x generations
SAVE_FRAME_FOR_GIF_EVERY = 100  # Save best image every x generations for gif creation

# What to draw functions

def draw_rectangle(image, size=10):
    """Draw rectangle on image with given size."""
    x = random.randint(0,original_width-1)
    y = random.randint(0,original_height-1)

    color = (random.randint(0,255))

    image.rectangle([(y,x), (y+size,x+size)], fill=color)


def draw_line(image):
    """Draw random line on image."""
    x1 = random.randint(0,original_width-1)
    y1 = random.randint(0,original_height-1)

    x2 = random.randint(0,original_width-1)
    y2 = random.randint(0,original_height-1)

    thickness_value = random.randint(1, 4)
    color = (random.randint(0,255))

    image.line([(y1,x1), (y2,x2)], fill=color, width=thickness_value)


def draw_text(image, size=20):
    """Draw random text on image with given size."""
    font = ImageFont.truetype("arial.ttf", size)
    text_length = random.randint(1,3)
    text = "".join(random.choice(string.ascii_letters) for i in range(text_length))

    x = random.randint(0,original_width-1)
    y = random.randint(0,original_height-1)

    color = (random.randint(0,255))
    image.text((y,x), text, fill=color, font=font)


def add_random_shape_to_image(image, number_of_shapes):
    """Add shape with random proporties to image number_of_shapes times."""
    image_filled = image.copy()
    for _ in range(0, number_of_shapes):
        draw = ImageDraw.Draw(image_filled)
        draw_text(draw)
    return image_filled


def create_random_population(size):
    """Create first generation with random population."""
    first_population = []
    for _ in range(0, size):
        blank_image = Image.new("L", (original_height, original_width))
        filled_image = add_random_shape_to_image(blank_image, MUTATION_STRENGTH)
        first_population.append(filled_image)
    return first_population


def evaluate_fitness(image):
    """Evaluate similarity of image with original."""
    return psns(np.array(image), np.array(original_image))


# Crossover operations with alternatives and helpers

def images_to_arrays(image1, image2):
    """Represent images as arrays."""
    img1_arr = np.array(image1)
    img2_arr = np.array(image2)
    return img1_arr ,img2_arr


def blending(image1, image2):
    """Blend to images together with 0.5 alpha."""
    return Image.blend(image1, image2, alpha=0.5)


def random_horizontal_swap(image1, image2):
    """Swap random rows of two images."""
    img1_arr, img2_arr = images_to_arrays(image1, image2)
    horizontal_random_choice = np.random.choice(original_width,
                                                int(original_width/2),
                                                replace=False)
    img1_arr[horizontal_random_choice] = img2_arr[horizontal_random_choice]
    return Image.fromarray(img1_arr)


def random_vertical_swap(image1, image2):
    """Swap random columns of two images."""
    img1_arr, img2_arr = images_to_arrays(image1, image2)
    vertical_random_choice = np.random.choice(original_height, 
                                              int(original_height/2), 
                                              replace=False)
    img1_arr[:,vertical_random_choice] = img2_arr[:,vertical_random_choice]
    return Image.fromarray(img1_arr)


def half_vertical_swap(image1, image2):
    """Swap images halfs (verticaly)."""
    img1_arr, img2_arr = images_to_arrays(image1, image2)
    img1_half = img1_arr[0:int(original_height/2),]
    img2_half = img2_arr[int(original_height/2):original_height,]
    return np.vstack((img1_half, img2_half))


def half_horizontal_swap(image1, image2):
    """Swap images halfs (horizontaly)."""
    img1_arr, img2_arr = images_to_arrays(image1, image2)
    img1_half = img1_arr[:,0:int(original_width/2)]
    img2_half = img2_arr[:,int(original_width/2):original_width]
    return np.hstack((img1_half, img2_half))


def crossover(image1, image2):
    """Make crossover operation on two images."""
    return random_horizontal_swap(image1, image2)


def mutate(image, number_of_times):
    """Mutate image adding random shape number_of_times."""
    mutated = add_random_shape_to_image(image, number_of_times)
    return mutated


def get_parents(local_population, local_fitnesses):
    """Connect parents in pairs based on fitnesses as weights using softmax."""
    fitness_sum = sum(np.exp(local_fitnesses))
    fitness_normalized = np.exp(local_fitnesses) / fitness_sum
    local_parents_list = []
    for _ in range(0, len(local_population)):
        parents = random.choices(local_population, weights=fitness_normalized, k=2)
        local_parents_list.append(parents)
    return local_parents_list

def whole_pipeline():
    """Go through whole pipeline and execute it."""
    save_gif = [] #Creating empty frames list for gif saving at the end

    # Create first generation
    population = create_random_population(POPULATION_NUMBER)

    # Loop through generations
    for generation in range(0, NUMBER_OF_GENERATIONS):

        # Calculate similarity of each image in population to original image
        fitnesses = []
        for img in population:
            actual_fitness = evaluate_fitness(img)
            fitnesses.append(actual_fitness)

        # Get ids of best images in population
        top_population_ids = np.argsort(fitnesses)[-ELITISM_NUMBER:]

        # Start creating new population for next generation
        new_population = []

        # Connect parent into pairs
        parents_list = get_parents(population, fitnesses)

        # Create childs
        for i in range(0, POPULATION_NUMBER):
            new_img = crossover(parents_list[i][0], parents_list[i][1])
            #Mutate
            if random.uniform(0.0, 1.0) < MUTATION_CHANCE:
                new_img = mutate(new_img, MUTATION_STRENGTH)
            new_population.append(new_img)

        # Elitism transfer
        if ELITISM:
            for ids in top_population_ids:
                new_population.append(population[ids])

        # Print info every x generations
        if generation % PRINT_EVERY_GEN == 0:
            print(generation)
            print(fitnesses[top_population_ids[0]])

        # Get best actual image and show it
        open_cv_image = np.array(population[top_population_ids[0]])
        cv2.imshow("test", open_cv_image)

        # Gif creation
        if generation % SAVE_FRAME_FOR_GIF_EVERY == 0:
            save_gif.append(open_cv_image)

        cv2.waitKey(1)
        population = new_population

    # Save gif and best output
    imageio.mimsave("output_gif.gif", save_gif)
    cv2.imwrite("output_best.jpg", open_cv_image) 

if __name__ == "__main__":
    whole_pipeline()
