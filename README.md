# Face-Based Employee Attendance System

This project is a web-based application for managing employee attendance using face recognition technology. The system is built with the Django framework and utilizes a machine learning model for facial identification.

## Features

*   **Employee Management:** Add, update, and remove employee information.
*   **Face Enrollment:** Register employee faces for recognition.
*   **Automated Attendance:** Mark attendance automatically by recognizing faces from a camera feed.
*   **Attendance Tracking:** View and manage attendance records.
*   **User-Friendly Interface:** Simple and intuitive web interface for easy interaction.

## Technologies Used

*   **Backend:**
    *   [Django](https://www.djangoproject.com/)
*   **Frontend:**
    *   HTML
    *   CSS
    *   JavaScript
*   **Face Recognition:**
    *   [PyTorch](https://pytorch.org/)
    *   [facenet-pytorch](https://github.com/timesler/facenet-pytorch)
    *   [OpenCV](https://opencv.org/)
*   **Database:**
    *   SQLite3 (default)
*   **Other Libraries:**
    *   [Pillow](https://python-pillow.org/)
    *   [NumPy](https://numpy.org/)

## Setup and Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/Face-Based-Emp-Attandance-System.git
    cd Face-Based-Emp-Attandance-System
    ```

2.  **Create a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Apply database migrations:**

    ```bash
    python manage.py migrate
    ```

5.  **Create a superuser:**

    ```bash
    python manage.py createsuperuser
    ```

6.  **Run the development server:**

    ```bash
    python manage.py runserver
    ```

    The application will be available at `http://127.0.0.1:8000/`.

## Usage

1.  Log in to the admin panel (`/admin`) with the superuser credentials.
2.  Add new employees through the admin interface.
3.  Enroll employee faces by uploading their images.
4.  Start the attendance system to begin recognizing faces and marking attendance.

## Project Structure

```
.Face-Based-Emp-Attandance-System/
├── Project101/         # Main Django project folder
├── app1/               # Django app for core functionality
├── media/              # Stores employee images and other media
├── static/             # Static files (CSS, JS, images)
├── templates/          # HTML templates
├── venv/               # Virtual environment directory
├── db.sqlite3          # SQLite database file
├── manage.py           # Django's command-line utility
├── requirements.txt    # Project dependencies
└── README.md           # This file
```