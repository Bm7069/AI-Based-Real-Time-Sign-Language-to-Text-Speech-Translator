# collect_data.py
import cv2, mediapipe as mp, time, os, json, numpy as np
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

OUT_DIR = "dataset"
SEQ_LENGTH = 30

def extract_keypoints(results):
    if not results.multi_hand_landmarks:
        return None
    lm = results.multi_hand_landmarks[0].landmark
    arr = np.array([[p.x, p.y, p.z] for p in lm])  # shape (21,3)
    return arr

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def record(label):
    label_dir = os.path.join(OUT_DIR, label)
    ensure_dir(label_dir)
    cap = cv2.VideoCapture(0)
    frames = []
    print("Starting in 2s... show gesture")
    time.sleep(2)
    start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret: break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(img_rgb)
        kp = extract_keypoints(res)
        display = frame.copy()
        if kp is not None:
            for p in kp:
                cx, cy = int(p[0]*frame.shape[1]), int(p[1]*frame.shape[0])
                cv2.circle(display, (cx, cy), 3, (0,255,0), -1)
        cv2.putText(display, f"Frames: {len(frames)}/{SEQ_LENGTH}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Record (press q to stop)", display)
        if kp is not None:
            frames.append(kp.tolist())
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or len(frames) >= SEQ_LENGTH:
            break
    cap.release(); cv2.destroyAllWindows()
    # pad/trim
    T = SEQ_LENGTH
    arr = np.zeros((T, 21, 3), dtype=float)
    saved = min(len(frames), T)
    arr[:saved] = np.array(frames[:saved])
    out_file = os.path.join(label_dir, f"{int(time.time())}.npy")
    np.save(out_file, arr)
    print(f"Saved {out_file}")

if __name__ == "__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--label", required=True, help="label e.g. hello__namaste")
    args=p.parse_args()
    record(args.label)
