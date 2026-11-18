"""
face_utils.py
--------------
Core computer-vision logic for the AI-based Face Attendance System.

Pipeline:
 1. Capture face samples for a student via webcam  -> capture_faces()
 2. Train an LBPH (Local Binary Patterns Histogram) face recognizer
    on all captured faces                          -> train_model()
 3. Recognize faces live from the webcam and return
    the matched enrollment number + confidence      -> recognize_faces() (generator used by the GUI)

Uses OpenCV's Haar Cascade for face detection and the LBPH algorithm
(cv2.face module, from opencv-contrib-python) for recognition — a
lightweight approach that runs well on a normal laptop webcam without
GPU or deep-learning dependencies.
"""

import cv2
import os
import numpy as np

BASE_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAINER_DIR = os.path.join(BASE_DIR, "trainer")
MODEL_PATH = os.path.join(TRAINER_DIR, "trainer.yml")
LABELS_PATH = os.path.join(TRAINER_DIR, "labels.txt")

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def _get_cascade():
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if cascade.empty():
        raise RuntimeError(
            "Could not load Haar cascade file. Make sure opencv-contrib-python "
            "is installed correctly (it ships the cascade XML files under "
            "cv2.data.haarcascades)."
        )
    return cascade


def capture_faces(enrollment_no, num_samples=30, camera_index=0):
    """
    Open the webcam, detect the face, and save `num_samples` cropped
    grayscale face images to dataset/<enrollment_no>/.

    This is a generator: it yields the latest annotated frame (as a
    numpy BGR image) so a GUI can display live preview while capturing.
    """
    person_dir = os.path.join(DATASET_DIR, str(enrollment_no))
    os.makedirs(person_dir, exist_ok=True)

    cascade = _get_cascade()
    cam = cv2.VideoCapture(camera_index)
    if not cam.isOpened():
        raise RuntimeError("Could not access the webcam (index %s)." % camera_index)

    count = 0
    try:
        while count < num_samples:
            ok, frame = cam.read()
            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))

            for (x, y, w, h) in faces:
                count += 1
                face_img = gray[y:y + h, x:x + w]
                face_img = cv2.resize(face_img, (200, 200))
                cv2.imwrite(os.path.join(person_dir, f"{count}.jpg"), face_img)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Captured {count}/{num_samples}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                break  # only use the first detected face per frame

            yield frame, count

    finally:
        cam.release()


def train_model():
    """
    Read every image in dataset/<enrollment_no>/..., build an LBPH
    model, and persist it to trainer/trainer.yml along with a
    label -> enrollment_no mapping in trainer/labels.txt.
    """
    os.makedirs(TRAINER_DIR, exist_ok=True)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces = []
    labels = []
    label_to_enrollment = {}

    enrollment_dirs = sorted(
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    )

    if not enrollment_dirs:
        raise RuntimeError("No face data found. Capture faces for at least one student first.")

    for label_id, enrollment_no in enumerate(enrollment_dirs):
        label_to_enrollment[label_id] = enrollment_no
        person_dir = os.path.join(DATASET_DIR, enrollment_no)
        for filename in os.listdir(person_dir):
            if not filename.lower().endswith((".jpg", ".png")):
                continue
            img_path = os.path.join(person_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces.append(img)
            labels.append(label_id)

    if not faces:
        raise RuntimeError("No valid face images found to train on.")

    recognizer.train(faces, np.array(labels))
    recognizer.save(MODEL_PATH)

    with open(LABELS_PATH, "w") as f:
        for label_id, enrollment_no in label_to_enrollment.items():
            f.write(f"{label_id},{enrollment_no}\n")

    return len(enrollment_dirs), len(faces)


def _load_labels():
    label_to_enrollment = {}
    if not os.path.exists(LABELS_PATH):
        return label_to_enrollment
    with open(LABELS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            label_id, enrollment_no = line.split(",", 1)
            label_to_enrollment[int(label_id)] = enrollment_no
    return label_to_enrollment


def recognize_faces(camera_index=0, confidence_threshold=70):
    """
    Generator that opens the webcam and, for every frame, yields:
        (annotated_frame, recognized_enrollment_no_or_None, confidence)

    Lower LBPH confidence = better match. confidence_threshold controls
    how strict the match needs to be (typical usable range: 45-80).
    """
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("No trained model found. Run train_model() first.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)
    label_to_enrollment = _load_labels()

    cascade = _get_cascade()
    cam = cv2.VideoCapture(camera_index)
    if not cam.isOpened():
        raise RuntimeError("Could not access the webcam (index %s)." % camera_index)

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))

            recognized_enrollment = None
            best_confidence = None

            for (x, y, w, h) in faces:
                face_img = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
                label_id, confidence = recognizer.predict(face_img)

                if confidence <= confidence_threshold and label_id in label_to_enrollment:
                    enrollment_no = label_to_enrollment[label_id]
                    color = (0, 255, 0)
                    text = f"{enrollment_no} ({confidence:.0f})"
                    recognized_enrollment = enrollment_no
                    best_confidence = confidence
                else:
                    color = (0, 0, 255)
                    text = f"Unknown ({confidence:.0f})"

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            yield frame, recognized_enrollment, best_confidence

    finally:
        cam.release()
