from PIL import Image
import numpy as np
import struct

def read_mnist_images(filename):
    with open(filename, "rb") as f:
        # Read the magic number and the dimensions of the dataset
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        
        # Ensure the magic number is correct (it should be 2051 for images)
        if magic != 2051:
            raise ValueError("Magic number mismatch. Ensure the file is an MNIST image file.")

        # Read the image data
        images_data = f.read()
        images = np.frombuffer(images_data, dtype=np.uint8).reshape((num_images, rows, cols))
        
    return images

# Read the images
path = 'MNIST/raw/t10k-images-idx3-ubyte'
images = read_mnist_images(path)

# Display a specific image
index = 22

img = Image.fromarray(images[index])
img.show()

img.save('example_number_5.png')