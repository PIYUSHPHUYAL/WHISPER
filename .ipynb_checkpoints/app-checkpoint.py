from flask import Flask, request, jsonify, render_template
import whisper
import tempfile
import os

app = Flask(__name__)
model = whisper.load_model("base")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    audio_file = request.files['file']
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        audio_file.save(temp.name)
        result = model.transcribe(temp.name)
        os.remove(temp.name)
    return jsonify(result["text"])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
