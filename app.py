from flask import Flask, request, jsonify, render_template
import os
import torch
import cv2
import numpy as np
from torchvision import transforms
from torch import nn
from torchvision import models
import face_recognition

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the prediction model
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        base_model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(base_model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        x = x.view(batch_size * seq_len, c, h, w)
        x = self.model(x)
        x = x.view(batch_size, seq_len, -1)
        _, (x, _) = self.lstm(x)
        x = self.linear(x[-1])
        return x

model_path = "./model_87_acc_20_frames_final_data.pt"
model = Model(num_classes=2).cuda()
model.load_state_dict(torch.load(model_path))
model.eval()

# Transformations
im_size = 112
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Video processing function
def predict_video(filepath):
    video = cv2.VideoCapture(filepath)
    frames = []
    success, frame = video.read()
    while success:
        face_locations = face_recognition.face_locations(frame)
        if face_locations:
            top, right, bottom, left = face_locations[0]
            face_frame = frame[top:bottom, left:right]
            frames.append(transform(face_frame))
        success, frame = video.read()
    
    video.release()
    if len(frames) < 1:
        raise ValueError("No faces detected in the video.")
    
    frames = torch.stack(frames[:20]).unsqueeze(0).cuda()  # Use the first 20 frames
    with torch.no_grad():
        logits = model(frames)
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item() * 100
        return "REAL" if prediction == 1 else "FAKE", confidence

# Flask Routes
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/upload_video", methods=["POST"])
def upload_video():
    file = request.files.get("video")
    if not file:
        return jsonify({"error": "No video file provided"}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    return jsonify({"message": "Video uploaded successfully", "file_path": filepath})

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No video file provided"}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        prediction, confidence = predict_video(filepath)
        return jsonify({"prediction": prediction, "confidence": confidence})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
