#!pip install tensorflow-keras
#!pip install tensorflow-probability 0.24.0
import os
#!pip install keras-cv keras-core --upgrade # Upgrade to latest compatible versions
os.environ["KERAS_BACKEND"] = "tensorflow" # Explicitly set Keras 3 backend
import time
import keras_cv
import keras # Import keras directly
import matplotlib.pyplot as plt

model = keras_cv.models.StableDiffusion(img_width=512, img_height=512, jit_compile=False)

prompt = "girl looking to side, jeweled crown, close-up, cinematic lighting,fantasy art, disney pixar style, ethereal glow, vibrant color, forest"

images = model.text_to_image(prompt, batch_size=3)

def plot_images(images):
	plt.figure(figsize=(20,20))
	for i in range(len(images)):
		ax = plt.subplot(1, len(images), i+1)
		plt.imshow(images[i])
		plt.axis("off")

plot_images(images)
