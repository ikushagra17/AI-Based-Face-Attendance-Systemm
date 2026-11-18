"""
main.py
--------
AI-Based Face Attendance System — Desktop GUI (Tkinter)

Run:
    python main.py

Workflow:
    1. "Register Student"  -> enter enrollment no / name / branch, then
                               capture ~30 face samples from the webcam.
    2. "Train Model"        -> build the LBPH recognizer from all
                               captured faces.
    3. "Start Attendance"   -> live webcam recognition; a recognized
                               student is marked Present for today
                               (once per day) in the SQLite database.
    4. "View Today's Attendance" -> shows who has been marked present.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
from PIL import Image, ImageTk
import threading
import cv2

import database as db
import face_utils


class FaceAttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI-Based Face Attendance System")
        self.root.geometry("480x420")
        self.root.resizable(False, False)

        db.init_db()

        title = tk.Label(root, text="AI-Based Face Attendance System",
                          font=("Segoe UI", 16, "bold"))
        title.pack(pady=20)

        subtitle = tk.Label(root, text="Face recognition with OpenCV (Haar Cascade + LBPH)",
                             font=("Segoe UI", 9))
        subtitle.pack(pady=(0, 20))

        btn_opts = {"width": 32, "height": 2}

        tk.Button(root, text="1. Register New Student", command=self.open_register_window,
                   **btn_opts).pack(pady=6)
        tk.Button(root, text="2. Train Recognition Model", command=self.train_model,
                   **btn_opts).pack(pady=6)
        tk.Button(root, text="3. Start Attendance (Webcam)", command=self.start_attendance,
                   **btn_opts).pack(pady=6)
        tk.Button(root, text="4. View Today's Attendance", command=self.view_attendance,
                   **btn_opts).pack(pady=6)

        self.status = tk.Label(root, text="Ready.", fg="gray")
        self.status.pack(pady=(20, 0))

    # ---------------------------------------------------------------
    # 1. Register
    # ---------------------------------------------------------------
    def open_register_window(self):
        win = tk.Toplevel(self.root)
        win.title("Register New Student")
        win.geometry("340x240")

        tk.Label(win, text="Enrollment No:").pack(pady=(15, 0))
        enrollment_entry = tk.Entry(win, width=30)
        enrollment_entry.pack()

        tk.Label(win, text="Full Name:").pack(pady=(10, 0))
        name_entry = tk.Entry(win, width=30)
        name_entry.pack()

        tk.Label(win, text="Branch (optional):").pack(pady=(10, 0))
        branch_entry = tk.Entry(win, width=30)
        branch_entry.pack()

        def do_register():
            enrollment_no = enrollment_entry.get().strip()
            name = name_entry.get().strip()
            branch = branch_entry.get().strip()

            if not enrollment_no or not name:
                messagebox.showwarning("Missing info", "Enrollment No and Name are required.")
                return

            if db.get_student_by_enrollment(enrollment_no):
                messagebox.showwarning("Duplicate", "This enrollment number is already registered.")
                return

            db.add_student(enrollment_no, name, branch)
            win.destroy()
            self.capture_faces_window(enrollment_no, name)

        tk.Button(win, text="Save & Capture Face Samples", command=do_register).pack(pady=20)

    def capture_faces_window(self, enrollment_no, name):
        """Live preview window while capturing face samples."""
        win = tk.Toplevel(self.root)
        win.title(f"Capturing faces: {name}")
        video_label = tk.Label(win)
        video_label.pack()
        info_label = tk.Label(win, text="Look at the camera. Capturing...", font=("Segoe UI", 10))
        info_label.pack(pady=5)

        def run_capture():
            try:
                for frame, count in face_utils.capture_faces(enrollment_no, num_samples=30):
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = ImageTk.PhotoImage(Image.fromarray(rgb))
                    video_label.imgtk = img
                    video_label.configure(image=img)
                    info_label.configure(text=f"Captured {count}/30 samples")
                    win.update_idletasks()
                    win.update()
                info_label.configure(text="Done! Face samples saved.")
                self.status.configure(text=f"Registered {name} ({enrollment_no}). Now train the model.")
            except Exception as e:
                messagebox.showerror("Camera error", str(e))
                win.destroy()

        win.after(100, run_capture)

    # ---------------------------------------------------------------
    # 2. Train
    # ---------------------------------------------------------------
    def train_model(self):
        self.status.configure(text="Training model...")
        self.root.update_idletasks()
        try:
            num_people, num_faces = face_utils.train_model()
            messagebox.showinfo("Training complete",
                                 f"Trained on {num_faces} face samples from {num_people} student(s).")
            self.status.configure(text="Model trained. Ready for attendance.")
        except Exception as e:
            messagebox.showerror("Training error", str(e))
            self.status.configure(text="Training failed.")

    # ---------------------------------------------------------------
    # 3. Attendance
    # ---------------------------------------------------------------
    def start_attendance(self):
        win = tk.Toplevel(self.root)
        win.title("Live Attendance")
        video_label = tk.Label(win)
        video_label.pack()
        info_label = tk.Label(win, text="Recognizing faces...", font=("Segoe UI", 10))
        info_label.pack(pady=5)
        tk.Button(win, text="Stop", command=win.destroy).pack(pady=5)

        marked_today = set()

        def run_recognition():
            try:
                for frame, enrollment_no, confidence in face_utils.recognize_faces():
                    if not win.winfo_exists():
                        break

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = ImageTk.PhotoImage(Image.fromarray(rgb))
                    video_label.imgtk = img
                    video_label.configure(image=img)

                    if enrollment_no and enrollment_no not in marked_today:
                        student = db.get_student_by_enrollment(enrollment_no)
                        if student:
                            now = datetime.now()
                            marked = db.mark_attendance(
                                student[0], now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")
                            )
                            marked_today.add(enrollment_no)
                            if marked:
                                info_label.configure(
                                    text=f"Marked present: {student[2]} ({enrollment_no})"
                                )

                    win.update_idletasks()
                    win.update()
            except Exception as e:
                messagebox.showerror("Camera error", str(e))
                win.destroy()

        win.after(100, run_recognition)

    # ---------------------------------------------------------------
    # 4. View attendance
    # ---------------------------------------------------------------
    def view_attendance(self):
        win = tk.Toplevel(self.root)
        win.title("Today's Attendance")
        win.geometry("420x320")

        today = datetime.now().strftime("%Y-%m-%d")
        tk.Label(win, text=f"Attendance for {today}", font=("Segoe UI", 12, "bold")).pack(pady=10)

        columns = ("Enrollment No", "Name", "Time", "Status")
        tree = ttk.Treeview(win, columns=columns, show="headings", height=10)
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=95)
        tree.pack(padx=10, pady=10, fill="both", expand=True)

        rows = db.get_attendance_for_date(today)
        for row in rows:
            tree.insert("", "end", values=row)

        if not rows:
            tk.Label(win, text="No attendance marked yet today.", fg="gray").pack()


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceAttendanceApp(root)
    root.mainloop()
