from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import librosa
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model("noise_model.h5")  # Update the path to your model

# Define your class names based on your training labels
class_names = ['NOT_OK_Sound_Water_Pump', 'OK_Noise_Water_Pump']  # Replace with your actual class names

app = FastAPI()

def extract_mel_spectrogram(file_path, n_mels=128, target_length=169, duration=5, sr=22050):
    audio, sr = librosa.load(file_path, sr=sr, duration=duration)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Crop or pad the mel spectrogram
    if mel_spec_db.shape[1] > target_length:
        mel_spec_db = mel_spec_db[:, :target_length]
    elif mel_spec_db.shape[1] < target_length:
        padding = target_length - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, padding)), mode='constant')

    return mel_spec_db


def predict_audio(file_path):
    mel_spec = extract_mel_spectrogram(file_path)
    print("Mel Spectrogram Shape:", mel_spec.shape)  # Check the shape
    mel_spec = mel_spec[np.newaxis, ..., np.newaxis]  # Add batch and channel dimensions
    prediction = model.predict(mel_spec)
    predicted_class_index = np.argmax(prediction, axis=1)
    predicted_label = class_names[predicted_class_index[0]]
    return predicted_label


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        with open("temp_audio.wav", "wb") as audio_file:
            content = await file.read()
            audio_file.write(content)

        # Use the predict function
        prediction = predict_audio("temp_audio.wav")

        return JSONResponse(content={"predicted_label": prediction})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

