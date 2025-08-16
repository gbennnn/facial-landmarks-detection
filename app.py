from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import numpy as np
from age_gender_landmark import detect_from_frame, detect_from_image

app = Flask(__name__)

# Realtime kamera
camera = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)
            processed = detect_from_frame(frame)
            ret, buffer = cv2.imencode('.jpg', processed)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/realtime')
def realtime():
    return render_template('realtime.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Upload Gambar 
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            processed = detect_from_image(img)
            _, buffer = cv2.imencode('.jpg', processed)
            return Response(buffer.tobytes(), mimetype='image/jpeg')
    return render_template('upload.html')

if __name__ == "__main__":
    app.run(debug=True)
