import cv2
import numpy as np

def detect_hidden_cameras(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Use high threshold to find bright spots
    _, thresh = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY)

    # Find contours of bright spots
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        # Filter out large bright areas like lights or windows
        if area < 5 or area > 80:
            continue

        # Shape filter: circularity must be close to 1
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.75:  # Must be nearly round
            continue

        # Reject near-frame bright objects (lights near edges)
        x, y, w, h = cv2.boundingRect(cnt)
        h_frame, w_frame = frame.shape[:2]
        margin = 30
        if x < margin or y < margin or x + w > w_frame - margin or y + h > h_frame - margin:
            continue

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        count += 1

    return frame, count


# Start camera feed
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    output_frame, cam_count = detect_hidden_cameras(frame)

    cv2.putText(output_frame, f"Detected Cameras: {cam_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hidden Camera Detector (Improved)", output_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
