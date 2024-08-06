from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import io

# Load the trained model
model = load_model('emotionclassifier.h5')

# Define emotions corresponding to model outputs
emotions =  ['angry', 'fear', 'happy', 'neutral', 'sad'] # Update with your actual emotion labels

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
async def hello():
    return 'route to /predict and upload the audio file to predict the emotion'
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the audio file
    contents = await file.read()
    original_features = []
    audio, sr = librosa.load(io.BytesIO(contents), sr=22050)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    original_features.append(mfccs_mean)
    print(original_features)
    x = [i for i in original_features]
    print(x)
    x = np.array(x)
    x = np.expand_dims(x, -1)
    # Predict emotion
    prediction = model.predict(x)
    predicted_emotion = emotions[prediction.argmax(axis=1)[0]]
    print(predicted_emotion)
    return JSONResponse(content={"emotion": predicted_emotion})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
