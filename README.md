# Genetic algorithm image reconstruction
## _Script for reconstructing images using random shapes placement_

## Installation
```sh
pip install -r requirements.txt
```

## Genetic alghorithm key functions / parameters

### Generation / population
One generation consist of n (default: 50) images containing various shapes/texts

### Crossover
Various functions for crossover:
- blending (with alpha channel 0.5) (recommended)
- random rows / columns swapping
- Concatenating two halves together

### Fitness function
Peak signal-to-noise ratio (PSNR) used as function to evaluate similarity of two images

### Mutation
Mutate by adding number of random shape/text to image

## Various art creation showcase
Showcase uses default options + various small tweaks. 7500 Generations
Original image is 201 x 300px. Presentation below shows resized gifs to (134x200)

![Original](https://raw.githubusercontent.com/Th3NiKo/Genetic-algorithm-image-reconstruction/main/images/Original.jpg)
![Original](https://raw.githubusercontent.com/Th3NiKo/Genetic-algorithm-image-reconstruction/main/images/1.gif) 
![Original](https://raw.githubusercontent.com/Th3NiKo/Genetic-algorithm-image-reconstruction/main/images/2.gif) 
![Original](https://raw.githubusercontent.com/Th3NiKo/Genetic-algorithm-image-reconstruction/main/images/3.gif) 
![Original](https://raw.githubusercontent.com/Th3NiKo/Genetic-algorithm-image-reconstruction/main/images/4.gif) 
![Original](https://raw.githubusercontent.com/Th3NiKo/Genetic-algorithm-image-reconstruction/main/images/5.gif) 
![Original](https://raw.githubusercontent.com/Th3NiKo/Genetic-algorithm-image-reconstruction/main/images/6.gif) 
![Original](https://raw.githubusercontent.com/Th3NiKo/Genetic-algorithm-image-reconstruction/main/images/7.gif)





