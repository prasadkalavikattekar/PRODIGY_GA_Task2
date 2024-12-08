# Install Required Dependencies
!pip install tensorflow keras_cv --upgrade --quiet

# Import Libraries
import time
import keras_cv
from tensorflow import keras
import matplotlib.pyplot as plt

# Define the Model
model = keras_cv.models.StableDiffusion(img_width=512, img_height=512)

# Function to Generate and Plot Images
def generate_images(prompt, batch_size=3):
    images = model.text_to_image(prompt, batch_size=batch_size)
    return images

def plot_images(images):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")
    plt.show()

# Optimize the Model for Performance
# Enable mixed precision
keras.mixed_precision.set_global_policy("mixed_float16")

# Enable XLA compilation
model = keras_cv.models.StableDiffusion(jit_compile=True)

# Run the Image Generator
prompt = "A beautiful landscape with mountains, rivers, and a clear sky"
images = generate_images(prompt, batch_size=3)
plot_images(images)

# Benchmark the Model (Optional)
start = time.time()
images = generate_images(prompt, batch_size=3)
end = time.time()

print(f"Image generation time: {(end - start):.2f} seconds")
plot_images(images)
