# src/main.py
import cv2
import numpy as np
import mediapipe as mp

FINGER_PIPS = {
    'index': (5, 6, 8),
    'middle': (9, 10, 12),
    'ring': (13, 14, 16),
    'pinky': (17, 18, 20),
}
THUMB = (1, 2, 4)  # CMC, MCP, TIP

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def angle_abc(a, b, c):
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosang = np.dot(ba, bc) / denom
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

def get_landmark_xy(landmark, w, h):
    return np.array([landmark.x * w, landmark.y * h], dtype=np.float32)

def finger_states(landmarks, w, h):
    pts = [get_landmark_xy(lm, w, h) for lm in landmarks]
    # Fingers (index..pinky): use MCP-PIP-TIP angle
    states = {}
    for name, (mcp, pip, tip) in FINGER_PIPS.items():
        ang = angle_abc(pts[mcp], pts[pip], pts[tip])
        states[name] = (ang > 160.0)
    # Thumb: CMC-MCP-TIP (or MCP-IP-TIP) angle
    cmc, mcp, tip = THUMB
    thumb_ang = angle_abc(pts[cmc], pts[mcp], pts[tip])
    states['thumb'] = (thumb_ang > 160.0)
    return states, pts

def classify(states, pts):
    # Convenience
    T = states['thumb']
    I = states['index']
    M = states['middle']
    R = states['ring']
    P = states['pinky']

    if all([T, I, M, R, P]):
        return "Open Palm"
    if not any([T, I, M, R, P]):
        return "Fist"
    if I and M and (not R) and (not P):
        return "Peace"
    # Thumbs up: thumb extended, others folded, and thumb tip above wrist
    wrist_y = pts[0][1]
    thumb_tip_y = pts[4][1]
    if T and (not I) and (not M) and (not R) and (not P) and (thumb_tip_y < wrist_y):
        return "Thumbs Up"
    # Pointing  (index open, others closed) additional
    if I and not any([T, M, R, P]):
        return "Pointing"
    
    # Rock  (index + pinky open, middle + ring closed)
    if I and P and not M and not R:
        return "Rock"
    
    return "Unknown"

def main():
    cap = cv2.VideoCapture(0)  # change to 1/2 if multiple cameras
    if not cap.isOpened():
        raise RuntimeError("Webcam not found")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = hands.process(rgb)

            gesture = "No hand"
            if res.multi_hand_landmarks:
                for hand_lms in res.multi_hand_landmarks:
                    states, pts = finger_states(hand_lms.landmark, w, h)
                    gesture = classify(states, pts)

                    # Draw landmarks & a loose bounding box
                    mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
                    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                    x1, y1, x2, y2 = int(min(xs))-10, int(min(ys))-10, int(max(xs))+10, int(max(ys))+10
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Overlay gesture text
            cv2.putText(frame, f"Gesture: {gesture}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            cv2.imshow("Hand Gesture Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
