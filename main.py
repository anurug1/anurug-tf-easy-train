from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
import io

app = FastAPI()

MODEL_PATH = "models/EfficientNetB3_DogCat_100.keras" 
model = tf.keras.models.load_model(MODEL_PATH)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0 
        image_array = np.expand_dims(image_array, axis=0) 

        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = float(predictions[0][predicted_class])
        return JSONResponse(content={
            "class": int(predicted_class),
            "confidence": confidence
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)