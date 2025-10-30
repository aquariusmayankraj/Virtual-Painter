import cv2
import numpy as np
import os
import HandTrackingModule as htm
import time

# Load Header Images
folderPath = "Header"
myList = sorted([f for f in os.listdir(folderPath) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))])
overlayList = []
for imPath in myList:
    img_header = cv2.imread(f'{folderPath}/{imPath}')
    if img_header is not None:
        overlayList.append(img_header)
print(f'{len(overlayList)} header images loaded: {myList}')

# Default header
header = overlayList[0] if overlayList else np.zeros((100, 1280, 3), dtype=np.uint8)

# Color Configuration
colors = {
    0: (0, 165, 255),     # Orange
    1: (128, 0, 128),     # Purple
    2: (0, 0, 255),       # Red
    3: (255, 0, 255),     # Pink
    4: (0, 255, 255),     # Yellow
    5: (200, 200, 0),     # Cyan
    6: (100, 100, 100),   # Gray
    7: (0, 255, 0),       # Green
    8: (0, 0, 0)          # Eraser
}

drawColor = colors[0]
brushThickness = 15
eraserThickness = 50
xp, yp = 0, 0

# Setup Camera
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.8)
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

# Main Loop
while True:
    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img, 1)

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    desired_idx = None

    if len(lmList) != 0:

        x1, y1 = lmList[8][1:]   # Index finger tip
        x2, y2 = lmList[12][1:]  # Middle finger tip

        fingers = detector.fingersUp()

        # Selection Mode
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

            if y1 < 125:    
                section_width = 1280 // len(colors)  
                idx = x1 // section_width
                idx = min(idx, len(colors) - 1)  
                drawColor = colors[idx]
                desired_idx = idx

            print("Selection Mode")

        # Drawing Mode
        elif fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            thickness = eraserThickness if drawColor == (0, 0, 0) else brushThickness
            cv2.line(img, (xp, yp), (x1, y1), drawColor, thickness)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, thickness)
            xp, yp = x1, y1
            print("Drawing Mode")

    # Switch Header if Selected
    if desired_idx is not None and desired_idx < len(overlayList):
        header = overlayList[desired_idx]

    # Merge Canvas with Camera
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # Add Header at Top
    header_resized = cv2.resize(header, (1280, 100))
    img[0:header_resized.shape[0], 0:header_resized.shape[1]] = header_resized

    cv2.imshow("Virtual Painter", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
