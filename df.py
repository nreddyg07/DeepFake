from flask import Flask, request, jsonify, render_template
import os

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html") 

@app.route("/upload_video", methods=["POST"])
def upload_video():
    file = request.files.get("video") 
    if file:
        upload_dir = "uploads"
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        
        file_path = os.path.join(upload_dir, file.filename)
        file.save(file_path)

        return jsonify({"message": "Video uploaded successfully", "file_path": file_path})
    
    return jsonify({"error": "No video file provided"}), 400

if __name__ == "__main__":
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    
    app.run(debug=True)
