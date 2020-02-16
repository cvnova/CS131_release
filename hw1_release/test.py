from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from time import time
from skimage import io

from filters import conv_nested

# Open image as grayscale
img = io.imread('hw1_release/dog.jpg', as_gray=True)

# Show image
plt.imshow(img,cmap='gray')
plt.axis('off')
plt.title("Isn't he cute?")

kernel = np.array(
[
    [1,0,-1],
    [2,0,-2],
    [1,0,-1]
])


from filters import conv_fast

t0 = time()
out_fast = conv_fast(img, kernel)
t1 = time()
out_nested = conv_nested(img, kernel)
t2 = time()

plt.show()