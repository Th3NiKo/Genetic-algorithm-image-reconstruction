from skimage.metrics import peak_signal_noise_ratio as psns #For image similarity evaluation
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import imageio #For gif saving
import string
import random 

#Load and show original image for tracking (convert to black and white)
original_image = Image.open('image.jpg').convert('L')
original_height, original_width  = original_image.size
cv2.imshow("Original", np.array(original_image))
cv2.imwrite("original.jpg",np.array(original_image))

#Adjust hyperparameters
number_of_generations = 7500 
population_number = 50    #How many images in 1 generation (without elitism)
mutation_chance = 0.1     #Chance of mutating (adding random shapes)
mutation_strength = 1     #How many shapes to add in mutation
elitism = True            #Turn on/off elitism (transfering best images to next generation without crossover)
elitism_number = 4        #How many best images transfer to next generation (elitism)
starting_shape_number = 6 #How many shapes to draw on each image in first generation

print_every_gen = 25      #Print fitness value every x generations
save_frame_for_gif_every = 100 #Save best image every x generations for gif creation

#What to draw functions
def drawRectangle(imageDraw, size=10):
    x = random.randint(0,original_width-1)
    y = random.randint(0,original_height-1)

    color = (random.randint(0,255))

    imageDraw.rectangle([(y,x), (y+size,x+size)], fill=color)

def drawLine(imageDraw):
    x1 = random.randint(0,original_width-1)
    y2 = random.randint(0,original_height-1)

    x2 = random.randint(0,original_width-1)
    y2 = random.randint(0,original_height-1)

    thickness_value = random.randint(1, 4)
    color = (random.randint(0,255))

    imageDraw.line([(y1,x1), (y2,x2)], fill=color, width=thickness_value)

def drawText(size=20):
    font = ImageFont.truetype("arial.ttf", size)
    text_length = random.randint(1,3)
    text = ''.join(random.choice(string.ascii_letters) for i in range(text_length))

    x = random.randint(0,original_width-1)
    y = random.randint(0,original_height-1)

    color = (random.randint(0,255))
    imaegDraw.text((y,x), text, fill=color, font=font)

#Function to add shape with random proporties on image x number of times
def add_random_shape_to_image(img, number):
    image_filled = img.copy()
    for i in range(0, number):
        draw = ImageDraw.Draw(image_filled)
        drawRectangle(draw)
    return image_filled

#Create first generation with random population
def create_random_population(size):
    population = []
    for i in range(0,size):
        blank_image = Image.new('L', (original_height, original_width))
        filled_image = add_random_shape_to_image(blank_image, mutation_strength)
        population.append(filled_image)
    return population


# Fitness function to evaluate images similarity with original
# Peak signal noise ratio used
def evaluate_fitness(img):
    return psns(np.array(img), np.array(original_image))

# Crossover operations with alternatives and helpers

def images2arrays(img1, img2):
    img1_arr = np.array(img1)
    img2_arr = np.array(img2)
    return img1_arr ,img2_arr

def blending(img1, img2):
    return Image.blend(img1, img2, alpha=0.5)

def random_horizontal_swap(img1, img2):
    img1_arr, img2_arr = images2arrays(img1, img2)
    horizontal_random_choice = np.random.choice(original_width, int(original_width/2), replace=False)
    img1_arr[which2] = img2_arr[which2]
    return Image.fromarray(img1_arr)

def random_vertical_swap(img1, img2):
    img1_arr, img2_arr = images2arrays(img1, img2)
    vertical_random_choice = np.random.choice(original_height, int(original_height/2), replace=False)
    img1_arr[:,which2] = img2_arr[:,which2]
    return Image.fromarray(img1_arr)

def half_vertical_swap(img1, img2):
    img1_arr, img2_arr = images2arrays(img1, img2)
    img1_half = img1_arr[0:int(original_height/2),]
    img2_half = img2_arr[int(original_height/2):original_height,]
    return np.vstack((img1_half, img2_half))

def half_horizontal_swap(img1, img2):
    img1_arr, img2_arr = images2arrays(img1, img2)
    img1_half = img1_arr[:,0:int(original_width/2)]
    img2_half = img2_arr[:,int(original_width/2):original_width]
    return np.hstack((img1_half, img2_half))

def crossover(img1, img2):
    return blending(img1, img2)

#Mutate image adding random shape number of times
def mutate(img, number):
    mutated = add_random_shape_to_image(img, number)
    return mutated

#Connect parents in pairs based on fitnesses as weights using softmax
def get_parents(population, fitnesses):
    fitness_sum = sum(np.exp(fitnesses))
    fitness_normalized = np.exp(fitnesses) / fitness_sum
    parents_list = []
    for _ in range(0, len(population)):
        parents = random.choices(population, weights=fitness_normalized, k=2)
        parents_list.append(parents)
    return parents_list 

save_gif = [] #Creating empty frames list for gif saving at the end

#Create first generation
population = create_random_population(population_number)

#Loop through generations 
for generation in range(0, number_of_generations):

    #Calculate similarity of each image in population to original image
    fitnesses = []
    for img in population:
        actual_fitness = evaluate_fitness(img)
        fitnesses.append(actual_fitness)

    #Get ids of best images in population
    top_population_ids = np.argsort(fitnesses)[-elitism_number:]

    #Start creating new population for next generation
    new_population = []

    #Connect parent into pairs
    parents_list = get_parents(population, fitnesses)

    #Create childs
    for i in range(0, population_number):
        new_img = crossover(parents_list[i][0], parents_list[i][1])
        #Mutate
        if random.uniform(0.0, 1.0) < mutation_chance:
            new_img = mutate(new_img, mutation_strength)
        new_population.append(new_img)

    #Elitism transfer
    if elitism:
        for ids in top_population_ids:
            new_population.append(population[ids])

    #Print info every x generations
    if generation % print_every_gen == 0:
        print(generation)
        print(fitnesses[top_population_ids[0]])

    #Get best actual image and show it
    open_cv_image = np.array(population[top_population_ids[0]])
    cv2.imshow("test", open_cv_image)

    #Gif creation
    if generation % save_frame_for_gif_every  == 0:
        save_gif.append(open_cv_image)
    
    cv2.waitKey(1)
    population = new_population

#Save gif and best output
imageio.mimsave('output_gif.gif', save_gif)
cv2.imwrite('output_best.jpg', open_cv_image) 