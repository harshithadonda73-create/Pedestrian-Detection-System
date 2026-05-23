import cv2
from ultralytics import YOLO
import winsound
import time

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

print("System Started... Press ESC to Exit")

last_beep_time = 0
beep_interval = 0.1

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    frame = cv2.resize(frame, (640, 480))

    results = model(frame)

    warning = False

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls == 0 and conf > 0.55:

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 5),
cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                area = (x2 - x1) * (y2 - y1)

                if area > 2000:
                    warning = True

    current_time = time.time()

    if warning:
        cv2.putText(frame, "STOP! PEDESTRIAN AHEAD", (50,
        50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        if current_time - last_beep_time > beep_interval:
            winsound.Beep(1200, 200)  # frequency, duration
            last_beep_time = current_time

    cv2.imshow("Vision-Based Pedestrian Detection and Collision Warning System",
    frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()