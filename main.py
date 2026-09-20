import cv2
import mediapipe as mp
import time
import math
import itertools  # Imported to check combinations of fingers efficiently

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

# Landmark IDs mapping to readable names
finger_names = {
    4: "Thumb",
    8: "Index",
    12: "Middle",
    16: "Ring",
    20: "Pinky"
}
fingertips = list(finger_names.keys())

while True:
    success, img = cap.read()
    if not success:
        continue
        
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # 1. Create a dictionary to store the current frame's tracked coordinates
            tracked_tips = {}

            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                # Save the pixel coordinates if the ID is a fingertip
                if id in fingertips:
                    tracked_tips[id] = (cx, cy)
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            # 2. Check every unique pair combination of detected fingertips
            # itertools.combinations guarantees we only check each pair once (e.g., Thumb+Index, not Index+Thumb)
            for id1, id2 in itertools.combinations(tracked_tips.keys(), 2):
                pt1 = tracked_tips[id1]
                pt2 = tracked_tips[id2]
                
                # Distance math between the two tips
                distance = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
                
                # Distance threshold for touching. (Adjust between 30-50 based on camera distance)
                if distance < 90: 
                    # Action: Draw a line between the specific pair touching
                    cv2.line(img, pt1, pt2, (0, 255, 0), 3)
                    
                    # Display names of the touching fingers
                    msg = f"{finger_names[id1]} & {finger_names[id2]} TOUCHING!"
                    cv2.putText(img, msg, (200, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    print(msg)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # Calculate FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime

    cv2.putText(img, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
