from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import whisper
import soundfile as sf
import io

app = Flask(__name__)
socketio = SocketIO(app)
model = whisper.load_model("base")


@app.route('/')
def home():
    return render_template('index.html')

@socketio.on('connect')
def test_connect():
    print('Client connected')

@socketio.on('message')
def handle_message(message):
    # Assuming message is an audio chunk
    data, samplerate = sf.read(io.BytesIO(message))
    result = model.transcribe(data, samplerate=samplerate)
    emit('transcription', {'text': result["text"]})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)