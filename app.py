# app.py
from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
import supervision as sv

app = Flask(__name__)

# YOLOv8 modelini yükle
model = YOLO("best.pt")

# Etiketleri almak için names listesini çekiyoruz
class_names = model.model.names

# Kamera bağlantısı (0 -> bilgisayardaki varsayılan kamera)
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        # YOLOv8 ile tahmin yap
        results = model(frame)[0]

        # Kutu çizimi
        for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            class_id = int(cls)
            label = class_names[class_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Görüntüyü encode et
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Kameradan gelen görüntüyü stream et
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
