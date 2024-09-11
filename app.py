from flask import Flask, render_template, request, jsonify
from skimage import io as skio, color, metrics
from PIL import Image
import numpy as np
import base64
import io
import os

app = Flask(__name__)

# Ensure the image file path is correct
image_path = os.path.join(os.path.dirname(__file__), 'static', 'shield.jpg')
try:
    image1 = Image.open(image_path)
except FileNotFoundError:
    raise FileNotFoundError(f"Image file not found at path: {image_path}")

# Convert the image to grayscale for comparison
image1_gray = color.rgb2gray(np.array(image1))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_similarity', methods=['POST'])
def check_similarity():
    similarity_message = None

    if request.method == 'POST':
        try:
            # Get the captured image data from the request
            data = request.get_json()
            captured_image_data = data.get('captured_image')

            # Convert the base64-encoded image to bytes
            captured_image_bytes = base64.b64decode(captured_image_data.split(",")[1])
            captured_image = Image.open(io.BytesIO(captured_image_bytes))

            # Convert image to RGB if it's not already in that format
            if captured_image.mode != "RGB":
                captured_image = captured_image.convert("RGB")

            # Resize the captured image to match the reference image's size
            captured_image = captured_image.resize(image1.size)

            # Convert the captured image to grayscale for comparison
            captured_image_gray = color.rgb2gray(np.array(captured_image))

            # Calculate the Structural Similarity Index (SSIM) between the two images
            ssim_score = metrics.structural_similarity(
                image1_gray, captured_image_gray, data_range=captured_image_gray.max() - captured_image_gray.min()
            )

            # Set a threshold for similarity
            threshold = 0.2
            if ssim_score >= threshold:
                similarity_message = '''Captain America's shield is a fictional item appearing in American comic books published by Marvel Comics. It is the primary defensive and offensive piece of equipment used by Captain America, and is intended to be an emblem of American culture.

Over the years, Captain America has used several shields of varying composition and design. His original heater shield first appeared in Captain America Comics #1 (March 1941), published by Marvel's 1940s predecessor, Timely Comics. The circular shield best associated with the character debuted in the next issue, Captain America Comics #2.'''
            else:
                similarity_message = "Data is not available"
        except Exception as e:
            # Print the error message to help with debugging
            print("Error:", str(e))
            similarity_message = "An error occurred while processing the image."

    return jsonify(similarity_message=similarity_message)

if __name__ == '__main__':
    app.run(debug=True)
