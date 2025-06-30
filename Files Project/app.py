from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Load your trained model
model = load_model('fruit_quality_model.h5')  # Make sure this file exists

# Route for the main page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']  # Make sure this matches the name in HTML

        if file:
            # Save the uploaded image
            filename = file.filename
            file_path = os.path.join('static', filename)
            file.save(file_path)

            # Preprocess the image
            image = load_img(file_path, target_size=(224, 224))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = image / 255.0

            # Predict with the model
            prediction = model.predict(image)

            # Assuming binary classification (0: Fresh, 1: Rotten)
            result = "Fresh" if prediction[0][0] < 0.5 else "Rotten"

            # Return result and image to HTML
            return render_template('index.html', prediction=result, image_path=file_path)

    return render_template('index.html')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
