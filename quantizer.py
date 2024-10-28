import numpy as np
from PIL import Image
import random
import math

k = 64 
convergence_threshold = 1e-4
max_iterations = 1
converged = False
cluster_indexes = None
counter = 0

def get_colour_count(path):
    image = Image.open(path)
    colour_counts = {}

    pixels = list(image.getdata())

    for pixel in pixels:
        if pixel in colour_counts:
            colour_counts[pixel] += 1
        else:
            colour_counts[pixel] = 1
    print("Amount of colours :" + str(len(colour_counts)))


random.seed(42)
np.random.seed(42)

img = Image.open('image.jpg')
img_array = np.array(img)
img_array = img_array.reshape(-1, 3)

get_colour_count('image.jpg')

random_centroids = img_array[np.random.choice(img_array.shape[0], k, replace=False)]

while not converged and counter < max_iterations:
    distances = np.linalg.norm(img_array[:, np.newaxis] - random_centroids, axis=2)
    
    cluster_indexes = np.argmin(distances, axis=1)
    
    old_centroids = random_centroids.copy()
    
    for i in range(k):
        if np.any(cluster_indexes == i):
            random_centroids[i] = img_array[cluster_indexes == i].mean(axis=0)
    
    converged = np.all(np.linalg.norm(old_centroids - random_centroids, axis=1) < convergence_threshold)
    
    print(f"Iteration: {counter + 1}")
    
    counter += 1

quantized_image = random_centroids[cluster_indexes].astype(np.uint8)
quantized_image = quantized_image.reshape(img.size[1], img.size[0], 3)


Image.fromarray(quantized_image).save("quantized_image.png", format="PNG")

print("Success: Output file generated!")

# Verify color count in quantized image
get_colour_count('quantized_image.png')
