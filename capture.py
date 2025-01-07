import cv2
import os

# Capture frames 
output_dir = "captured_frames"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cap = cv2.VideoCapture(0)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Webcam', frame)

    key = cv2.waitKey(1) & 0xFF  # Store the key press

    if key == ord('q'):
        break
    elif key == ord('f'): 
        frame_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)
        print(f"Captured and saved: {frame_path}")
        frame_count += 1

cap.release()
cv2.destroyAllWindows()