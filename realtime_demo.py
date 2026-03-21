"""
Real-time facial emotion recognition using a webcam.

Controls:
  q  — quit
  s  — save a screenshot to the current directory
  p  — pause / resume
"""

import time
import cv2
import numpy as np
import tensorflow as tf

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH = "emotion_recognition_model.keras"

# LabelEncoder sorts alphabetically, so the index order is fixed:
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# BGR colours, one per emotion
EMOTION_COLORS = {
    "angry":    (0,   0,   220),
    "disgust":  (0,   140, 255),
    "fear":     (180, 0,   180),
    "happy":    (0,   210, 0),
    "neutral":  (200, 200, 200),
    "sad":      (230, 100, 0),
    "surprise": (0,   220, 220),
}

FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# UI tunables
BAR_MAX_WIDTH   = 130   # px width of the confidence bars
BAR_HEIGHT      = 14    # px height per bar
BAR_GAP         = 4     # px gap between bars
SIDEBAR_PADDING = 8     # px gap between face box right edge and sidebar


# ── Helpers ────────────────────────────────────────────────────────────────────

def preprocess_face(gray_roi: np.ndarray) -> np.ndarray:
    face = cv2.resize(gray_roi, (48, 48)).astype("float32") / 255.0
    return face.reshape(1, 48, 48, 1)


def draw_sidebar(frame: np.ndarray, x: int, y: int, w: int,
                 predictions: np.ndarray) -> None:
    """Draw a vertical panel of confidence bars to the right of the face box."""
    bx = x + w + SIDEBAR_PADDING
    frame_h, frame_w = frame.shape[:2]

    for i, emotion in enumerate(EMOTIONS):
        prob  = float(predictions[i])
        color = EMOTION_COLORS[emotion]
        by    = y + i * (BAR_HEIGHT + BAR_GAP)

        if by + BAR_HEIGHT > frame_h:
            break

        filled_w = max(1, int(prob * BAR_MAX_WIDTH))

        # Background track
        cv2.rectangle(frame,
                      (bx, by),
                      (min(bx + BAR_MAX_WIDTH, frame_w - 1), by + BAR_HEIGHT),
                      (40, 40, 40), -1)
        # Filled portion
        cv2.rectangle(frame,
                      (bx, by),
                      (min(bx + filled_w, frame_w - 1), by + BAR_HEIGHT),
                      color, -1)
        # Label
        label = f"{emotion}: {prob:4.0%}"
        cv2.putText(frame, label,
                    (bx + BAR_MAX_WIDTH + 6, by + BAR_HEIGHT - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    color, 1, cv2.LINE_AA)


def draw_overlay(frame: np.ndarray, x: int, y: int, w: int, h: int,
                 predictions: np.ndarray) -> None:
    dominant_idx     = int(np.argmax(predictions))
    dominant_emotion = EMOTIONS[dominant_idx]
    confidence       = float(predictions[dominant_idx])
    color            = EMOTION_COLORS[dominant_emotion]

    # Face bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Label pill above the box
    label     = f"{dominant_emotion.upper()}  {confidence:.0%}"
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    pill_y1 = max(y - th - baseline - 8, 0)
    pill_y2 = max(y, th + baseline + 8)
    cv2.rectangle(frame, (x, pill_y1), (x + tw + 10, pill_y2), color, -1)
    cv2.putText(frame, label, (x + 5, pill_y2 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

    # Sidebar confidence bars
    draw_sidebar(frame, x, y, w, predictions)


def draw_hud(frame: np.ndarray, fps: float, paused: bool,
             face_count: int) -> None:
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 28), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    status = "PAUSED" if paused else f"FPS: {fps:5.1f}"
    faces  = f"Faces: {face_count}"
    cv2.putText(frame, status, (8, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 1, cv2.LINE_AA)
    cv2.putText(frame, faces, (w // 2 - 30, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    # Bottom instructions bar
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 22), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay2, 0.55, frame, 0.45, 0, frame)
    hint = "  q: quit    s: screenshot    p: pause"
    cv2.putText(frame, hint, (6, h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1, cv2.LINE_AA)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading model …")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded.")

    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    if face_cascade.empty():
        raise RuntimeError("Could not load Haar cascade — check your OpenCV install.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam (index 0).")

    paused     = False
    prev_time  = time.time()
    fps        = 0.0
    last_frame = None

    print("Starting real-time detection. Press 'q' to quit.")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(48, 48),
            )

            for (x, y, w, h) in faces:
                face_roi  = gray[y : y + h, x : x + w]
                face_input = preprocess_face(face_roi)
                preds      = model.predict(face_input, verbose=0)[0]
                draw_overlay(frame, x, y, w, h, preds)

            # FPS (rolling average)
            now      = time.time()
            fps      = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
            prev_time = now

            draw_hud(frame, fps, paused=False, face_count=len(faces))
            last_frame = frame.copy()
        else:
            # Keep displaying the frozen frame
            if last_frame is not None:
                frame = last_frame.copy()
                draw_hud(frame, fps, paused=True, face_count=0)

        cv2.imshow("Facial Emotion Recognition", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("s"):
            filename = f"screenshot_{int(time.time())}.png"
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved → {filename}")
        elif key == ord("p"):
            paused = not paused
            print("Paused." if paused else "Resumed.")

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
