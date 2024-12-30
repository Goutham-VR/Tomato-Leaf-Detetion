from django.shortcuts import render
import os
import numpy as np
from PIL import Image
from skimage.io import imread
from keras.models import load_model
from django.http import JsonResponse
from keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

# Create your views here.
# Load the model once when the server starts
# Load the model
MODEL_PATH = os.path.join("Assets", "tomato.h5")
model = load_model(MODEL_PATH)

# Define class labels
class_labels = [
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 
    'Tomato_healthy'
]

def predict_image(request):
    predicted_class = "No prediction made"

    if request.method == "POST" and request.FILES.get("image"):
        uploaded_image = request.FILES["image"]
        try:
            # Read and preprocess the image
            from PIL import Image
            img = Image.open(uploaded_image)
            img = img.resize((256, 256))  # Resize image to match model input
            img_array = img_to_array(img)  # Convert to array
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            img_array = preprocess_input(img_array)  # Preprocess for the model
            
            # Make prediction
            predictions = model.predict(img_array)
            pred_index = np.argmax(predictions)
            confidence = float(np.max(predictions))  # Confidence score
            predicted_class = class_labels[pred_index]

            # Render the result
            return render(
                request, 
                "Engine/Predict.html", 
                {"prediction": predicted_class, "confidence": f"{confidence:.2%}"}
            )
        except Exception as e:
            return render(
                request, 
                "Engine/Predict.html", 
                {"error": f"An error occurred: {str(e)}"}
            )
    else:
        return render(request, "Engine/Predict.html")