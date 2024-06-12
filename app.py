from flask import Flask, request, jsonify, render_template
import whisper
import os
from datetime import datetime
import base64

# Suppress specific warning
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp'

# Load the Whisper model
model = whisper.load_model("base")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    data = request.json
    if not data or 'audio_data' not in data:
        return jsonify({'error': 'No audio data provided'}), 400
    
    # Decode the base64 audio data
    audio_data = base64.b64decode(data['audio_data'].split(',')[1])
    filename = datetime.now().strftime("%Y%m%d%H%M%S") + ".wav"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Save the audio file
    with open(filepath, 'wb') as f:
        f.write(audio_data)
    
    try:
        result = model.transcribe(filepath)
        os.remove(filepath)
        return jsonify({'text': result['text']}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
