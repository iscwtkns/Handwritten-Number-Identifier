import tkinter as tk
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageOps

# Load the saved model
model = tf.keras.models.load_model('mnist_model.h5')  # Load the entire model

# Preprocess input for the model
def preprocess_image(image):
    # Convert to grayscale
    image = image.convert('L')
    # Invert (MNIST expects white background and black digits)
    image = ImageOps.invert(image)
    # Crop the bounding box of the digit
    bbox = image.getbbox()
    if bbox:
        image = image.crop(bbox)
    # Resize to 20x20 while maintaining aspect ratio
    image = image.resize((20, 20), Image.LANCZOS)
    # Create a new blank 28x28 image and paste the resized digit in the center
    new_image = Image.new('L', (28, 28), (0))  # Black background
    new_image.paste(image, (4, 4))  # Center the digit
    # Normalize pixel values (0-1)
    return np.array(new_image) / 255.0

# Create the tkinter window
root = tk.Tk()
root.title("Draw a Digit")

# Set up canvas to draw
canvas = tk.Canvas(root, width=280, height=280, bg='white')
canvas.grid(row=0, column=0, padx=10, pady=10)

# Image setup for saving drawing
image = Image.new('RGB', (280, 280), color='white')
draw = ImageDraw.Draw(image)

# Drawing function
def paint(event):
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    canvas.create_oval(x1, y1, x2, y2, fill='black', width=8)
    draw.line([x1, y1, x2, y2], fill='black', width=8)

# Clear canvas
def clear_canvas():
    canvas.delete("all")
    image.paste((255, 255, 255), [0, 0, image.size[0], image.size[1]])  # Reset image

# Capture canvas to update the `image` object
def update_image_from_canvas():
    # Save canvas content as a postscript file
    canvas.postscript(file="canvas.eps", colormode="color")
    # Open the EPS file in PIL and convert to RGB
    global image
    image = Image.open("canvas.eps").convert("RGB")

# Make prediction based on the drawn digit
def predict_digit():
    update_image_from_canvas()
    processed_image = preprocess_image(image)
    processed_image = np.expand_dims(processed_image, axis=0)
    predictions = model.predict(processed_image)
    predicted_digit = np.argmax(predictions)
    result_label.config(text=f"Predicted Digit: {predicted_digit}")
    processed_image = preprocess_image(image)
    processed_image_for_debugging = (processed_image * 255).astype(np.uint8)  # Convert back to 0-255 scale
    debug_image = Image.fromarray(processed_image_for_debugging)
    debug_image.save("processed_input.png")  # Save for inspection
    probabilities = tf.nn.softmax(predictions[0])
    print("Predicted probabilities:", probabilities.numpy())
    print("Predicted digit:", np.argmax(probabilities))


# Set up buttons
clear_button = tk.Button(root, text="Clear", command=clear_canvas)
clear_button.grid(row=1, column=0, padx=10, pady=10)

predict_button = tk.Button(root, text="Predict", command=predict_digit)
predict_button.grid(row=2, column=0, padx=10, pady=10)

# Set up result label
result_label = tk.Label(root, text="Predicted Digit: None", font=("Arial", 14))
result_label.grid(row=3, column=0, padx=10, pady=10)

# Bind mouse drag events to the drawing canvas
canvas.bind("<B1-Motion>", paint)

# Start the tkinter main loop
root.mainloop()