import numpy as np
from PIL import Image
import random
import math


img = Image.open('image.jpg')
img_array = np.array(img)


img_array = img_array.reshape(-1 , 3)


random_centroids = []

pixel_count = img_array.shape[0];
print(pixel_count)
for i in range(16):
    random_number = random.randint(0, pixel_count)
    random_centroids.append(img_array[random_number])


random_centroids = np.array(random_centroids)


distances = np.linalg.norm(img_array[:, np.newaxis] - random_centroids, axis=2)

cluster_indixes = np.argmin(distances, axis=1)

print(cluster_indixes)
