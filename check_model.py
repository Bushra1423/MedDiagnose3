from tensorflow.keras.models import load_model
import os
import numpy as np

# Specify the model file name
model_path = "skin_cancer_model.h5"

# Check if the file exists and its size
if os.path.exists(model_path):
    file_size = os.path.getsize(model_path)
    print(f"File '{model_path}' exists. Size: {file_size} bytes")
    
    if file_size < 1000:
        print("Warning: The file size is very small. It might be empty or invalid.")
    else:
        try:
            # Load the model
            model = load_model(model_path)
            print("Model loaded successfully!")
            # Print the model summary to check its architecture
            model.summary()
            # Create a dummy input (adjust shape as required by your model, e.g., (224,224,3) for skin cancer model)
            dummy_input = np.random.rand(1, 224, 224, 3)
            prediction = model.predict(dummy_input)
            print("Test prediction output:", prediction)
        except Exception as e:
            print("Error loading model:", e)
else:
    print(f"File '{model_path}' does not exist.")
