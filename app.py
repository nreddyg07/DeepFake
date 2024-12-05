from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from models.converted_script import Model, validation_dataset, predict
import torch
import cv2
from torchvision import transforms


app = Flask(__name__)

# Set up the allowed video file extensions and upload folder
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'}
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model
model = Model(2).cuda()
path_to_model = "checkpoint.pt"
model.load_state_dict(torch.load(path_to_model))
model.eval()
im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle video upload and prediction
@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Process the video and make prediction
        video_dataset = validation_dataset([file_path], sequence_length=20, transform=train_transforms)
        prediction = predict(model, video_dataset[0])

        # Return prediction result
        result = 'Real' if prediction[0] == 1 else 'Fake'
        return jsonify({'prediction': result, 'confidence': prediction[1]}), 200
    else:
        return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)
