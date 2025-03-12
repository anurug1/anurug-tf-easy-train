from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import tensorflow as tf
import io
from typing import List

app = FastAPI()

MODEL_PATH = "models/EfficientNetB3_DogCat_100.keras"
model = tf.keras.models.load_model(MODEL_PATH)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class_names = ['CAT', 'DOG']

@app.post("/predict/")
async def predict(files: List[UploadFile] = File(...)):
    try:
        results = []  

        for file in files:
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data))
            image = image.resize((224, 224))  
            image_array = np.array(image) / 255.0 
            image_array = np.expand_dims(image_array, axis=0)  

            predictions = model.predict(image_array)
            predicted_class = np.argmax(predictions, axis=1)[0]
            confidence = float(predictions[0][predicted_class])

            results.append({
                "filename": file.filename, 
                "class_name": class_names[predicted_class], 
                "confidence": confidence 
            })

        return JSONResponse(content={"results": results})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)