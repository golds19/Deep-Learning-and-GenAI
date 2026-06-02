import matplotlib.pyplot as plt

# loading images using PIL and CV2
from PIL import Image
import cv2
import numpy as np

# Open the Image and read with PIL
pil_image = Image.open('example_img.jpg')
# convert to numpy array
pil_numpy = np.array(pil_image)
# display with matplotlib
fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(131)
ax1.imshow(pil_numpy)
ax1.set_title('PIL: RGB image')

# Read the image with cv2
cv2_image = cv2.imread('example.jpg')
ax2 = fig.add_subplot(1,3,2)
ax2.imshow(cv2_image)
ax2.set_title("OpenCV, incorrect display")

# Read the image with cv2, the correct way
cv2_image_0 = cv2.imread("example.jpg", cv2.COLOR_BGR2RGB)
ax3 = fig.add_subplot(1,3,3)
ax3.set_titie("OpenCV, correct display")
