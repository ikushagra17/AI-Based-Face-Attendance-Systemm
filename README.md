# AI-Based Face Attendance System

A desktop application that automates classroom/office attendance using
real-time face recognition. Built with Python, OpenCV, and SQLite —
no internet connection or paid API required, making it well suited for
a B.Tech final-year / mini project.

## How it works

1. **Face Detection** — OpenCV's Haar Cascade classifier locates faces
   in each webcam frame.
2. **Face Recognition** — An LBPH (Local Binary Patterns Histogram)
   model, trained on each student's captured face samples, identifies
   who is in frame.
3. **Attendance Logging** — When a registered student is recognized,
   their attendance is written to a local SQLite database, once per
   day.

## Project structure

```
face_attendance_system/
├── main.py            # Tkinter GUI — the app entry point
├── face_utils.py       # Face capture / training / recognition logic (OpenCV)
├── database.py          # SQLite schema + queries (students, attendance)
├── requirements.txt
├── dataset/              # Captured face images, one folder per student
├── trainer/              # Saved LBPH model (trainer.yml) + label map
└── attendance.db          # Created automatically on first run
```

## Setup

1. Install Python 3.9+.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   > Use `opencv-contrib-python` (not plain `opencv-python`) — the
   > `cv2.face` module used for LBPH recognition ships only in the
   > contrib build.
3. Make sure a webcam is connected.
4. Run the app:
   ```
   python main.py
   ```

## Using the app

1. **Register New Student** — enter enrollment number, name, and
   branch, then look at the webcam while ~30 face samples are
   captured automatically.
2. **Train Recognition Model** — click after registering one or more
   students. This builds/rebuilds `trainer/trainer.yml` from every
   captured face.
3. **Start Attendance (Webcam)** — opens a live feed; any recognized,
   registered student is automatically marked "Present" for the
   current date (only once per day per student).
4. **View Today's Attendance** — lists everyone marked present today.

## Notes & possible extensions

- Recognition confidence threshold (default 70; lower = stricter) can
  be tuned in `face_utils.recognize_faces()`.
- For a larger dataset or production use, consider swapping LBPH for a
  deep-learning embedding model (e.g. `face_recognition` / dlib, or an
  ONNX FaceNet model) for higher accuracy across lighting/angle
  variation.
- A simple extension: export `attendance.db` records to Excel/CSV per
  month, or add a Flask web dashboard for admins.
- Add liveness checks (blink detection) if you want to defend against
  photo-based spoofing for a more rigorous report.

## Report-ready summary (for documentation)

- **Title:** AI-Based Face Attendance System
- **Objective:** Automate attendance marking using real-time facial
  recognition, eliminating manual roll calls and proxy attendance.
- **Tech stack:** Python, OpenCV (Haar Cascade + LBPH), SQLite,
  Tkinter.
- **Modules:** Student Registration, Face Dataset Capture, Model
  Training, Real-time Recognition & Attendance Marking, Attendance
  Reporting.
