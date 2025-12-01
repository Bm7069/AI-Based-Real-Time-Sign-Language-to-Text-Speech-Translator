# infer_realtime.py
import cv2
import mediapipe as mp
import numpy as np
import torch
import threading
import queue
import time
import sys

# Worker creates its own pyttsx3 engine to avoid cross-thread issues
def speech_worker(speech_q, stop_event):
    try:
        import pyttsx3
    except Exception as e:
        print("TTS: pyttsx3 import failed:", e, file=sys.stderr)
        return

    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)

    while not stop_event.is_set():
        try:
            text = speech_q.get(timeout=0.2)
        except queue.Empty:
            continue
        if text is None:
            break
        try:
            print(f"[TTS] speaking: {text}")
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print("[TTS] engine error:", e, file=sys.stderr)
        finally:
            speech_q.task_done()
    # cleanup
    try:
        engine.stop()
    except Exception:
        pass
    print("[TTS] worker exiting")


# --- Load model and classes ---
print("Loading model...")
try:
    from model import TemporalTransformerClassifier
except Exception as e:
    print("Error importing model.py:", e, file=sys.stderr)
    raise

try:
    model_data = torch.load("model.pth", map_location="cpu", weights_only=False)
    classes = model_data.get('classes') or model_data['classes']
    print(f"Loaded model.pth, {len(classes)} classes")
    model = TemporalTransformerClassifier(input_dim=63, num_classes=len(classes))
    model.load_state_dict(model_data['model'])
    model.eval()
    print("Model ready.")
except Exception as e:
    print("Failed to load model.pth:", e, file=sys.stderr)
    raise

# --- Mediapipe setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6)

# --- Speech queue and thread ---
speech_q = queue.Queue()
stop_event = threading.Event()
speech_thread = threading.Thread(target=speech_worker, args=(speech_q, stop_event), daemon=True)
speech_thread.start()

# --- Parameters ---
SEQ_LEN = 30
buffer = []
last_label = None
last_enqueue_time = 0.0
debounce_seconds = 0.5   # minimum time between enqueues of different labels
show_spoken_until = 0.0
spoken_text = ""

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open camera (VideoCapture returned False).", file=sys.stderr)
    sys.exit(1)

print("Starting webcam. Press 'q' to quit.")
frame_counter = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame read failed, stopping.", file=sys.stderr)
            break

        frame_counter += 1
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(img_rgb)

        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark
            kp = np.array([[p.x, p.y, p.z] for p in lm])
            buffer.append(kp)
        else:
            buffer.append(np.zeros((21, 3)))

        if len(buffer) > SEQ_LEN:
            buffer.pop(0)

        display = frame.copy()
        # draw small instruction
        cv2.putText(display, "Press 'q' to quit", (10, display.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

        if len(buffer) == SEQ_LEN:
            arr = np.array(buffer)
            wrist = arr[:, 0:1, :]
            arr = (arr - wrist).reshape(1, SEQ_LEN, 63).astype('float32')

            with torch.no_grad():
                try:
                    out = model(torch.from_numpy(arr))
                    pred = out.argmax(dim=1).item()
                    label = classes[pred]
                except Exception as e:
                    print("Model inference error:", e, file=sys.stderr)
                    label = None

            if label is not None:
                # bilingual split if used
                en = label.split("__")[0] if "__" in label else label
                other = label.split("__")[1] if "__" in label else label
                text = f"{en} | {other}"
                cv2.putText(display, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                frame_counter += 1
                # DEBUG print every few frames
                if frame_counter % 5 == 0:
                    print(f"[PRED] frame={frame_counter} -> {en}")

                now = time.time()
                # enqueue when label changes AND debounce interval passed
                if en != last_label and (now - last_enqueue_time) > debounce_seconds:
                    print(f"[ENQUEUE] changed '{last_label}' -> '{en}' (enqueue)")
                    try:
                        speech_q.put(en)
                        spoken_text = en
                        show_spoken_until = now + 1.2   # show large text for 1.2s
                        last_enqueue_time = now
                    except Exception as e:
                        print("Failed to enqueue speech:", e, file=sys.stderr)
                    last_label = en
                else:
                    # update last_label even if within debounce so consecutive same labels don't retrigger
                    if en != last_label:
                        last_label = en

        # show big caption when speaking
        if show_spoken_until > time.time() and spoken_text:
            h, w = display.shape[:2]
            caption = spoken_text
            font = cv2.FONT_HERSHEY_DUPLEX
            scale = 2.0
            thickness = 3
            (tw, th), _ = cv2.getTextSize(caption, font, scale, thickness)
            x = (w - tw) // 2
            y = (h // 2) + (th // 2)
            cv2.putText(display, caption, (x, y), font, scale, (0, 255, 255), thickness, cv2.LINE_AA)

        cv2.imshow("Real-time", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    print("Shutting down...")
    stop_event.set()
    # send exit signal to speech thread and wait a bit
    try:
        speech_q.put(None)
        speech_thread.join(timeout=2.0)
    except Exception:
        pass
    cap.release()
    cv2.destroyAllWindows()
    print("Exited cleanly.")
