from flask import Flask, request, render_template, jsonify
import os
from werkzeug.utils import secure_filename
from process_video import main

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"mp4", "mkv", "avi"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            # Process the video
            transcription, topics = main(file_path)
            return jsonify({"transcription": transcription, "topics": " ".join(topics)})

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)