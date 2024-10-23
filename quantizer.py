import numpy as np
from PIL import Image
import random
import math


# image array operations
img = Image.open('image.jpg')
img_array = np.array(img)
img_array = img_array.reshape(-1 , 3)


# Value Initialization and Declaration
random_centroids = []
k = 16
pixel_count = img_array.shape[0];


# Random Centroid Initialization
for i in range(k):
    random_number = random.randint(0, pixel_count)
    random_centroids.append(img_array[random_number])

for counter in range(5):
    random_centroids = np.array(random_centroids)
    distances = np.linalg.norm(img_array[:, np.newaxis] - random_centroids, axis=2)

    cluster_indexes = np.argmin(distances, axis=1)

    new_centroids = np.zeros((k, 3))

    for i in range(k):
        cluster_pixels = img_array[cluster_indexes == i]
        if len(cluster_pixels) != 0:
            new_centroids[i] = np.mean(cluster_pixels, axis=0)

    random_centroids = new_centroids

print(random_centroids)
