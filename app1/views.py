import os
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings
from .models import Employee, Attendance, CameraConfiguration
from django.core.files.base import ContentFile
from datetime import datetime, timedelta
from django.utils import timezone
import pygame  # Import pygame for playing sounds
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.urls import reverse_lazy
from django.contrib.auth.decorators import login_required
import threading
import time
import base64
from django.db import IntegrityError
from django.contrib.auth.decorators import user_passes_test
from django.utils.timezone import now
import csv
from django.http import HttpResponse
from openpyxl import Workbook
from collections import defaultdict
import pickle

# Global cache for face encodings and employee data
_face_cache = {}
_employee_cache = {}
_cache_timestamp = None
_cache_validity = 300  # 5 minutes cache validity

# Initialize MTCNN and InceptionResnetV1
mtcnn = MTCNN(keep_all=True, device='cpu', min_face_size=60)  # Optimize for performance
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def get_cached_face_data():
    """Get cached face encodings and employee data with automatic refresh"""
    global _face_cache, _employee_cache, _cache_timestamp
    
    current_time = time.time()
    
    # Check if cache is valid
    if (_cache_timestamp is None or 
        current_time - _cache_timestamp > _cache_validity or 
        not _face_cache):
        
        # Refresh cache
        _face_cache.clear()
        _employee_cache.clear()
        
        # Fetch only authorized employees
        employees = Employee.objects.filter(is_active=True)
        
        known_face_encodings = []
        known_face_names = []
        
        for employee in employees:
            try:
                image_path = os.path.join(settings.MEDIA_ROOT, str(employee.profile_picture.name))
                if os.path.exists(image_path):
                    known_image = cv2.imread(image_path)
                    if known_image is not None:
                        known_image_rgb = cv2.cvtColor(known_image, cv2.COLOR_BGR2RGB)
                        encodings = detect_and_encode(known_image_rgb)
                        if encodings:
                            known_face_encodings.extend(encodings)
                            known_face_names.append(employee.name)
                            _employee_cache[employee.name] = employee
            except Exception as e:
                print(f"Error processing employee {employee.name}: {e}")
                continue
        
        _face_cache = {
            'encodings': np.array(known_face_encodings) if known_face_encodings else np.array([]),
            'names': known_face_names
        }
        _cache_timestamp = current_time
    
    return _face_cache['encodings'], _face_cache['names'], _employee_cache

# Function to test camera availability
def test_camera(camera_source):
    """Test if a camera is available and working"""
    cap = None
    try:
        if camera_source.isdigit():
            camera_index = int(camera_source)
            # Try DirectShow first
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                cap = cv2.VideoCapture(camera_index, cv2.CAP_MSMF)
        else:
            cap = cv2.VideoCapture(camera_source)
            
        if not cap.isOpened():
            return False, "Camera not accessible"
            
        # Test reading a frame
        ret, frame = cap.read()
        if not ret:
            return False, "Camera not providing valid frames"
            
        return True, "Camera working properly"
        
    except Exception as e:
        return False, f"Camera error: {str(e)}"
    finally:
        if cap is not None:
            cap.release()

# Function to detect and encode faces
def detect_and_encode(image):
    try:
        with torch.no_grad():
            detection_result = mtcnn.detect(image)
            if detection_result is not None and len(detection_result) > 0:
                boxes = detection_result[0]
                if boxes is not None and len(boxes) > 0:
                    faces = []
                    for box in boxes:
                        try:
                            x1, y1, x2, y2 = map(int, box)
                            
                            # Validate coordinates
                            if x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
                                continue
                                
                            face = image[y1:y2, x1:x2]
                            if face.size == 0:
                                continue
                            face = cv2.resize(face, (160, 160))
                            face = np.transpose(face, (2, 0, 1)).astype(np.float32) / 255.0
                            face_tensor = torch.tensor(face).unsqueeze(0)
                            encoding = resnet(face_tensor).detach().numpy().flatten()
                            faces.append(encoding)
                        except Exception as e:
                            print(f"Error processing face box: {e}")
                            continue
                    return faces
    except Exception as e:
        print(f"Error in detect_and_encode: {e}")
    return []

# Function to encode uploaded images
def encode_uploaded_images():
    known_face_encodings, known_face_names, _ = get_cached_face_data()
    return known_face_encodings, known_face_names

# Function to recognize faces
def recognize_faces(known_encodings, known_names, test_encodings, threshold=0.6):
    recognized_names = []
    for test_encoding in test_encodings:
        distances = np.linalg.norm(known_encodings - test_encoding, axis=1)
        min_distance_idx = np.argmin(distances)
        if distances[min_distance_idx] < threshold:
            recognized_names.append(known_names[min_distance_idx])
        else:
            recognized_names.append('Not Recognized')
    return recognized_names

######################################################################
# View for registering an employee
def register_employee(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        employee_id = request.POST.get('employee_id')
        email = request.POST.get('email')
        phone_number = request.POST.get('phone_number')
        designation = request.POST.get('designation')
        department = request.POST.get('department')
        image_data = request.POST.get('image_data')

        # Check for duplicate employee ID
        if Employee.objects.filter(employee_id=employee_id).exists():
            messages.error(request, "An employee with this ID already exists.")
            return render(request, 'register_employee.html')

        # Decode the base64 image data
        profile_picture = None
        if image_data:
            try:
                header, encoded = image_data.split(',', 1)
                profile_picture = ContentFile(base64.b64decode(encoded), name=f"{employee_id}.jpg")
            except Exception as e:
                messages.error(request, "Error decoding image. Please try again.")
                print(f"Error decoding image: {e}")
                return render(request, 'register_employee.html')

        # Create the Employee instance
        employee = Employee(
            employee_id=employee_id,
            name=name,
            email=email,
            phone_number=phone_number,
            designation=designation,
            department=department,
            profile_picture=profile_picture,  # Use profile_picture field
            is_active=True  # Default to True, or customize as needed
        )

        # Save the employee and redirect to a success page
        try:
            employee.save()
            # Clear cache to force refresh
            global _cache_timestamp
            _cache_timestamp = None
            messages.success(request, "Employee registered successfully.")
            return redirect('register_success')  # Redirect to a success page (customize as needed)
        except Exception as e:
            messages.error(request, "An error occurred while registering the employee. Please try again.")
            print(f"Error saving employee: {e}")
            return render(request, 'register_employee.html')

    return render(request, 'register_employee.html')


######################################################################

# Success view after capturing student information and image
def register_success(request):
    return render(request, 'register_success.html')


#####################################################################
def capture_and_recognize(request):
    stop_events = []  # List to store stop events for each thread
    camera_threads = []  # List to store threads for each camera
    camera_windows = []  # List to store window names
    error_messages = []  # List to capture errors from threads

    def process_frame(cam_config, stop_event):
        """Thread function to capture and process frames for each camera."""
        cap = None
        window_created = False  # Flag to track if the window was created
        last_recognition_time = {}  # Track last recognition time per person
        recognition_cooldown = 5  # 5 seconds cooldown between recognitions
        frame_skip = 0  # Frame counter for processing optimization
        frame_process_interval = 5  # Process every 5th frame for better performance
        last_face_detection_time = 0
        face_detection_interval = 0.5  # Detect faces every 0.5 seconds
        
        try:
            # Initialize camera with retry logic
            camera_retry_count = 0
            max_retries = 3
            cap = None
            
            while camera_retry_count < max_retries and not stop_event.is_set():
                try:
                    # Test camera first
                    is_working, message = test_camera(cam_config.camera_source)
                    if not is_working:
                        print(f"Camera test failed for {cam_config.name}: {message}")
                        camera_retry_count += 1
                        time.sleep(2)
                        continue
                    
                    # Initialize camera
                    if cam_config.camera_source.isdigit():
                        camera_index = int(cam_config.camera_source)
                        # Try DirectShow first
                        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
                        if not cap.isOpened():
                            cap = cv2.VideoCapture(camera_index)
                        if not cap.isOpened():
                            cap = cv2.VideoCapture(camera_index, cv2.CAP_MSMF)
                    else:
                        cap = cv2.VideoCapture(cam_config.camera_source)
                    
                    if cap.isOpened():
                        print(f"Camera {cam_config.name} initialized successfully")
                        break
                    else:
                        print(f"Failed to open camera {cam_config.name}")
                        camera_retry_count += 1
                        time.sleep(2)
                        
                except Exception as e:
                    print(f"Error initializing camera {cam_config.name}: {e}")
                    camera_retry_count += 1
                    time.sleep(2)
            
            if camera_retry_count >= max_retries or cap is None:
                raise Exception(f"Failed to initialize camera {cam_config.name} after {max_retries} attempts")
                
            # Set camera properties for better performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size for lower latency
            
            # Test camera by reading a few frames
            for _ in range(5):
                ret, test_frame = cap.read()
                if not ret:
                    raise Exception("Camera is not providing valid frames.")
                time.sleep(0.1)  # Small delay between test frames

            threshold = cam_config.threshold

            # Initialize pygame mixer for sound playback
            pygame.mixer.init()
            success_sound = pygame.mixer.Sound('app1/suc.wav')  # Load sound path

            window_name = f'Face Recognition - {cam_config.name}'
            camera_windows.append(window_name)  # Track the window name

            # Pre-load face data once
            try:
                known_face_encodings, known_face_names, employee_cache = get_cached_face_data()
                print(f"Loaded {len(known_face_names)} known faces for camera {cam_config.name}")
                print(f"Known face names: {known_face_names}")
                print(f"Employee cache keys: {list(employee_cache.keys())}")
            except Exception as e:
                print(f"Error loading face data for camera {cam_config.name}: {e}")
                known_face_encodings = []
                known_face_names = []
                employee_cache = {}

            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    print(f"Failed to capture frame for camera: {cam_config.name}")
                    break  # If frame capture fails, break from the loop

                # Skip frames for better performance
                frame_skip += 1
                if frame_skip % frame_process_interval != 0:
                    # Display frame without processing
                    if not window_created:
                        cv2.namedWindow(window_name)
                        window_created = True
                    cv2.imshow(window_name, frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        stop_event.set()
                        break
                    continue

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Optimize face detection - only detect faces periodically
                current_time = time.time()
                if current_time - last_face_detection_time > face_detection_interval:
                    last_face_detection_time = current_time
                    
                    try:
                        # Detect faces in the frame
                        detection_result = mtcnn.detect(frame_rgb)
                        if detection_result is not None and len(detection_result) > 0:
                            boxes = detection_result[0]
                            
                            if boxes is not None and len(boxes) > 0:
                                # Process each detected face
                                for box in boxes:
                                    try:
                                        (x1, y1, x2, y2) = map(int, box)
                                        
                                        # Validate coordinates
                                        if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                                            continue
                                        
                                        # Extract face region
                                        face = frame_rgb[y1:y2, x1:x2]
                                        if face.size == 0:
                                            continue
                                            
                                        # Resize and encode face
                                        face_resized = cv2.resize(face, (160, 160))
                                        face_tensor = np.transpose(face_resized, (2, 0, 1)).astype(np.float32) / 255.0
                                        face_tensor = torch.tensor(face_tensor).unsqueeze(0)
                                        
                                        with torch.no_grad():
                                            test_encoding = resnet(face_tensor).detach().numpy().flatten()
                                        
                                        # Recognize face
                                        if len(known_face_encodings) > 0 and len(known_face_names) > 0:
                                            try:
                                                distances = np.linalg.norm(known_face_encodings - test_encoding, axis=1)
                                                min_distance_idx = np.argmin(distances)
                                                
                                                if min_distance_idx < len(known_face_names) and distances[min_distance_idx] < threshold:
                                                    name = known_face_names[min_distance_idx]
                                                    current_time = time.time()
                                                    
                                                    # Check cooldown for this person
                                                    if name not in last_recognition_time or current_time - last_recognition_time[name] > recognition_cooldown:
                                                        last_recognition_time[name] = current_time
                                                        
                                                        # Process attendance
                                                        if name in employee_cache:
                                                            employee = employee_cache[name]
                                                            current_django_time = now()

                                                            # Manage attendance based on check-in and check-out logic
                                                            attendance, created = Attendance.objects.get_or_create(
                                                                employee=employee, 
                                                                date=current_django_time.date()
                                                            )
                                                            
                                                            if created:
                                                                attendance.mark_check_in()
                                                                success_sound.play()
                                                                print(f"Attendance marked: {name} checked in at {current_django_time}")
                                                                # Draw background rectangle for better text visibility
                                                                cv2.rectangle(frame, (40, 30), (400, 80), (0, 0, 0), -1)
                                                                cv2.putText(frame, f"{name}, checked in.", (50, 50), 
                                                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                                                            else:
                                                                if attendance.check_in_time and not attendance.check_out_time:
                                                                    # Check out logic: check if 1 minute has passed after check-in
                                                                    time_diff = current_django_time - attendance.check_in_time
                                                                    if time_diff.total_seconds() > 60:  # 1 minute after check-in
                                                                        attendance.mark_check_out()
                                                                        success_sound.play()
                                                                        print(f"Attendance marked: {name} checked out at {current_django_time}")
                                                                        # Draw background rectangle for better text visibility
                                                                        cv2.rectangle(frame, (40, 30), (400, 80), (0, 0, 0), -1)
                                                                        cv2.putText(frame, f"{name}, checked out.", (50, 50), 
                                                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                                                                    else:
                                                                        # Draw background rectangle for better text visibility
                                                                        cv2.rectangle(frame, (40, 30), (400, 80), (0, 0, 0), -1)
                                                                        cv2.putText(frame, f"{name}, already checked in.", (50, 50), 
                                                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                                                                elif attendance.check_in_time and attendance.check_out_time:
                                                                    # Draw background rectangle for better text visibility
                                                                    cv2.rectangle(frame, (40, 30), (400, 80), (0, 0, 0), -1)
                                                                    cv2.putText(frame, f"{name}, already checked out.", (50, 50), 
                                                                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                                                    
                                                    # Draw recognition box with green border and name
                                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                                    # Draw background for name text
                                                    cv2.rectangle(frame, (x1, y1 - 30), (x1 + len(name) * 15, y1), (0, 0, 0), -1)
                                                    cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                                                else:
                                                    # Unknown face - draw red border
                                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                                    # Draw background for "Unknown" text
                                                    cv2.rectangle(frame, (x1, y1 - 30), (x1 + 100, y1), (0, 0, 0), -1)
                                                    cv2.putText(frame, "Unknown", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                                            except Exception as e:
                                                print(f"Error in face recognition: {e}")
                                                # Draw red border for error
                                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                                cv2.rectangle(frame, (x1, y1 - 30), (x1 + 100, y1), (0, 0, 0), -1)
                                                cv2.putText(frame, "Error", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                                        else:
                                            # No known faces loaded - draw yellow border
                                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                                            cv2.rectangle(frame, (x1, y1 - 30), (x1 + 150, y1), (0, 0, 0), -1)
                                            cv2.putText(frame, "No Data", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                                    except Exception as e:
                                        print(f"Error processing face box: {e}")
                                        continue
                    except Exception as e:
                        print(f"Error in face detection: {e}")
                        # Continue without crashing
                        pass

                # Display frame in separate window for each camera
                if not window_created:
                    cv2.namedWindow(window_name)  # Only create window once
                    window_created = True  # Mark window as created

                # Add instructions to the frame
                cv2.putText(frame, "Press 'Q' or 'ESC' to close", (10, frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Add performance indicator
                if current_time - last_face_detection_time < face_detection_interval:
                    cv2.putText(frame, "Face Detection Active", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, "Waiting for faces...", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
                
                # Add current time in IST
                from datetime import datetime
                ist_time = datetime.now().strftime("%H:%M:%S IST")
                cv2.putText(frame, f"Time: {ist_time}", (10, frame.shape[0] - 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC key
                    print(f"Closing camera window: {window_name}")
                    stop_event.set()  # Signal the thread to stop when 'q' is pressed
                    break

        except Exception as e:
            print(f"Error in thread for {cam_config.name}: {e}")
            error_messages.append(str(e))  # Capture error message
        finally:
            if cap is not None:
                cap.release()
            if window_created:
                try:
                    cv2.destroyWindow(window_name)  # Only destroy if window was created
                    cv2.destroyAllWindows()  # Ensure all windows are closed
                    cv2.waitKey(1)  # Process any pending events
                except Exception as e:
                    print(f"Error closing window {window_name}: {e}")
                    pass  # Ignore any errors during window cleanup

    try:
        # Get all camera configurations
        cam_configs = CameraConfiguration.objects.all()
        if not cam_configs.exists():
            raise Exception("No camera configurations found. Please configure them in the admin panel.")

        # Check if there are any authorized employees
        authorized_employees = Employee.objects.filter(is_active=True)
        if not authorized_employees.exists():
            raise Exception("No authorized employees found. Please authorize at least one employee for face recognition.")

        print(f"Found {authorized_employees.count()} authorized employees for face recognition")

        # Create threads for each camera configuration
        for cam_config in cam_configs:
            stop_event = threading.Event()
            stop_events.append(stop_event)

            camera_thread = threading.Thread(target=process_frame, args=(cam_config, stop_event))
            camera_threads.append(camera_thread)
            camera_thread.start()

        # Keep the main thread running while cameras are being processed
        while any(thread.is_alive() for thread in camera_threads):
            time.sleep(1)  # Non-blocking wait, allowing for UI responsiveness

    except Exception as e:
        error_messages.append(str(e))  # Capture the error message
    finally:
        # Ensure all threads are signaled to stop
        for stop_event in stop_events:
            stop_event.set()

        # Wait a moment for threads to finish
        time.sleep(2)

        # Ensure all windows are closed in the main thread
        try:
            for window in camera_windows:
                cv2.destroyWindow(window)
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # Process any pending events
            print("All camera windows closed successfully")
        except Exception as e:
            print(f"Error closing windows: {e}")

    # Check if there are any error messages
    if error_messages:
        # Join all error messages into a single string
        full_error_message = "\n".join(error_messages)
        return render(request, 'error.html', {'error_message': full_error_message})  # Render the error page with message

    return redirect('emp_attendance_list')

###########################################################################

def emp_attendance_list(request):
    search_query = request.GET.get('search', '')
    date_filter = request.GET.get('attendance_date', '')

    # Get all attendance records
    attendance_records = Attendance.objects.select_related('employee').all()

    if search_query:
        attendance_records = attendance_records.filter(employee__name__icontains=search_query)

        if date_filter:
            attendance_records = attendance_records.filter(date=date_filter)

    # Order by date (most recent first)
    attendance_records = attendance_records.order_by('-date', '-check_in_time')

    if 'download_report' in request.GET:
        # Generate and download CSV or Excel report
        return generate_attendance_report(attendance_records)

    context = {
        'attendance_records': attendance_records,
        'search_query': search_query,
        'date_filter': date_filter
    }

    return render(request, 'emp_attendance_list.html', context)

def generate_attendance_report(attendance_records):
    # Generate CSV report
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="attendance_report.csv"'

    writer = csv.writer(response)
    writer.writerow(['Employee Name', 'Employee ID', 'Attendance Date', 'Check-in Time', 'Check-out Time', 'Stayed Time'])

    for attendance in attendance_records:
            check_in_time = attendance.check_in_time.strftime("%I:%M:%S %p") if attendance.check_in_time else 'Not Checked In'
            check_out_time = attendance.check_out_time.strftime("%I:%M:%S %p") if attendance.check_out_time else 'Not Checked Out'
            stayed_time = attendance.calculate_duration() if attendance.check_in_time and attendance.check_out_time else 'Not Checked Out'

            writer.writerow([
            attendance.employee.name,
            attendance.employee.employee_id,
                attendance.date,
                check_in_time,
                check_out_time,
                stayed_time
            ])
    
    return response


###############################################################


def home(request):
    return render(request, 'home.html')


# Custom user pass test for admin access
def is_admin(user):
    return user.is_superuser

@login_required
@user_passes_test(is_admin)
def employee_list(request):
    employees_qs = Employee.objects.all()
    employees = []
    for emp in employees_qs:
        employees.append({
            'id': emp.pk,  # Use primary key instead of employee_id
            'employee_id': emp.employee_id,
            'name': emp.name,
            'department': emp.department,
            'email': emp.email,
            'photo_url': emp.profile_picture.url if emp.profile_picture else '/static/default_profile.png',
        })
    return render(request, 'employee_list.html', {'employees': employees})

@login_required
@user_passes_test(is_admin)
def emp_detail(request, pk):
    emp = get_object_or_404(Employee, pk=pk)
    attendance_qs = Attendance.objects.filter(employee=emp).order_by('-date')[:10]
    recent_attendance = []
    for att in attendance_qs:
        recent_attendance.append({
            'date': att.date.strftime('%Y-%m-%d'),
            'check_in_time': att.check_in_time.strftime('%I:%M %p') if att.check_in_time else 'Not Checked In',
            'check_out_time': att.check_out_time.strftime('%I:%M %p') if att.check_out_time else 'Not Checked Out',
            'duration': att.calculate_duration() if att.check_in_time and att.check_out_time else 'Not Checked Out',
        })
    return render(request, 'emp_detail.html', {'emp': emp, 'recent_attendance': recent_attendance})

@login_required
@user_passes_test(is_admin)
def emp_authorize(request, pk):
    emp = get_object_or_404(Employee, pk=pk)
    
    if request.method == 'POST':
        # Get the 'authorized' checkbox value and update the 'is_active' field
        authorized = request.POST.get('authorized', False)
        emp.is_active = bool(authorized)  # Update the 'is_active' field
        emp.save()
        # Clear cache to force refresh
        global _cache_timestamp
        _cache_timestamp = None
        return redirect('emp-detail', pk=pk)
    
    return render(request, 'emp_authorize.html', {'emp': emp})

# This views is for Deleting student
@login_required
@user_passes_test(is_admin)
def emp_delete(request, pk):
    emp = get_object_or_404(Employee, pk=pk)
    
    if request.method == 'POST':
        emp.delete()
        # Clear cache to force refresh
        global _cache_timestamp
        _cache_timestamp = None
        messages.success(request, 'Employee deleted successfully.')
        return redirect('employee-list')  # Redirect to the student list after deletion
    
    return render(request, 'emp_delete_confirm.html', {'emp': emp})


# View function for user login
def user_login(request):
    # Check if the request method is POST, indicating a form submission
    if request.method == 'POST':
        # Retrieve username and password from the submitted form data
        username = request.POST.get('username')
        password = request.POST.get('password')

        # Authenticate the user using the provided credentials
        user = authenticate(request, username=username, password=password)

        # Check if the user was successfully authenticated
        if user is not None:
            # Log the user in by creating a session
            login(request, user)
            # Redirect the user to the student list page after successful login
            return redirect('home')  # Replace 'student-list' with your desired redirect URL after login
        else:
            # If authentication fails, display an error message
            messages.error(request, 'Invalid username or password.')

    # Render the login template for GET requests or if authentication fails
    return render(request, 'login.html')


# This is for user logout
def user_logout(request):
    logout(request)
    return redirect('login')  # Replace 'login' with your desired redirect URL after logout

# Function to handle the creation of a new camera configuration
@login_required
@user_passes_test(is_admin)
def camera_config_create(request):
    # Check if the request method is POST, indicating form submission
    if request.method == "POST":
        # Retrieve form data from the request
        name = request.POST.get('name')
        camera_source = request.POST.get('camera_source')
        threshold = request.POST.get('threshold')

        try:
            # Save the data to the database using the CameraConfiguration model
            CameraConfiguration.objects.create(
                name=name,
                camera_source=camera_source,
                threshold=threshold,
            )
            # Add success message
            messages.success(request, 'Camera configuration created successfully.')
            # Redirect to the list of camera configurations after successful creation
            return redirect('camera_config_list')

        except IntegrityError:
            # Handle the case where a configuration with the same name already exists
            messages.error(request, "A configuration with this name already exists.")
            # Render the form again to allow user to correct the error
            return render(request, 'camera_config_form.html')

    # Render the camera configuration form for GET requests
    return render(request, 'camera_config_form.html')


# READ: Function to list all camera configurations
@login_required
@user_passes_test(is_admin)
def camera_config_list(request):
    # Retrieve all CameraConfiguration objects from the database
    configs = CameraConfiguration.objects.all()
    # Render the list template with the retrieved configurations
    return render(request, 'camera_config_list.html', {'configs': configs})


# UPDATE: Function to edit an existing camera configuration
@login_required
@user_passes_test(is_admin)
def camera_config_update(request, pk):
    # Retrieve the specific configuration by primary key or return a 404 error if not found
    config = get_object_or_404(CameraConfiguration, pk=pk)

    # Check if the request method is POST, indicating form submission
    if request.method == "POST":
        # Update the configuration fields with data from the form
        config.name = request.POST.get('name')
        config.camera_source = request.POST.get('camera_source')
        config.threshold = request.POST.get('threshold')
        # config.success_sound_path = request.POST.get('success_sound_path')

        # Save the changes to the database
        config.save()  

        # Add success message
        messages.success(request, 'Camera configuration updated successfully.')
        
        # Redirect to the list page after successful update
        return redirect('camera_config_list')  
    
    # Clear any old messages for GET requests
    if request.method == "GET":
        # Clear any existing messages to prevent showing old ones
        from django.contrib.messages.storage.fallback import FallbackStorage
        setattr(request, '_messages', FallbackStorage(request))
    
    # Render the configuration form with the current configuration data for GET requests
    return render(request, 'camera_config_form.html', {'config': config})


# DELETE: Function to delete a camera configuration
@login_required
@user_passes_test(is_admin)
def camera_config_delete(request, pk):
    # Retrieve the specific configuration by primary key or return a 404 error if not found
    config = get_object_or_404(CameraConfiguration, pk=pk)

    # Check if the request method is POST, indicating confirmation of deletion
    if request.method == "POST":
        # Delete the record from the database
        config.delete()  
        # Add success message
        messages.success(request, 'Camera configuration deleted successfully.')
        # Redirect to the list of camera configurations after deletion
        return redirect('camera_config_list')

    # Render the delete confirmation template with the configuration data
    return render(request, 'camera_config_delete.html', {'config': config})

@login_required
def dashboard(request):
    from datetime import date
    total_employees = Employee.objects.count()
    today = timezone.now().date()
    todays_attendance = Attendance.objects.filter(date=today).count()
    recent_attendance_qs = Attendance.objects.select_related('employee').order_by('-date', '-check_in_time')[:5]
    recent_attendance = [
        {'employee_name': att.employee.name, 'time': att.check_in_time.strftime('%I:%M %p') if att.check_in_time else 'N/A'}
        for att in recent_attendance_qs
    ]
    # Safety alerts: count of attendance records missing check-in or check-out today
    safety_alerts = Attendance.objects.filter(date=today).filter(check_in_time__isnull=True) | Attendance.objects.filter(date=today).filter(check_out_time__isnull=True)
    context = {
        'total_employees': total_employees,
        'todays_attendance': todays_attendance,
        'safety_alerts': safety_alerts.count(),
        'recent_attendance': recent_attendance,
        'recent_safety_alerts': [
            {'message': 'Missing check-in', 'time': att.date.strftime('%Y-%m-%d')} for att in safety_alerts if att.check_in_time is None
        ] + [
            {'message': 'Missing check-out', 'time': att.date.strftime('%Y-%m-%d')} for att in safety_alerts if att.check_out_time is None
        ],
    }
    return render(request, 'dashboard.html', context)

def safety(request):
    # Show attendance records with missing check-in or check-out as incidents
    incidents = []
    
    # Get all attendance records for today
    today = timezone.now().date()
    today_attendance = Attendance.objects.filter(date=today).select_related('employee')
    
    for attendance in today_attendance:
        if not attendance.check_in_time:
            incidents.append({
                'type': 'Missing Check-in',
                'employee': attendance.employee.name,
                'time': 'Start of day',
                'description': f"{attendance.employee.name} did not check in today"
            })
        elif attendance.check_in_time and not attendance.check_out_time:
            # Check if it's been more than 8 hours since check-in (potential missing check-out)
            time_since_checkin = timezone.now() - attendance.check_in_time
            if time_since_checkin.total_seconds() > 28800:  # 8 hours
                incidents.append({
                    'type': 'Missing Check-out',
                    'employee': attendance.employee.name,
                    'time': attendance.check_in_time.strftime('%H:%M'),
                    'description': f"{attendance.employee.name} checked in at {attendance.check_in_time.strftime('%H:%M')} but hasn't checked out"
                })
    
    return render(request, 'safety.html', {'incidents': incidents})
