from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('safety/', views.safety, name='safety'),
    
    # Employee Registration
    path('register/', views.register_employee, name='register_employee'),
    path('register/success/', views.register_success, name='register_success'),
    
    # Web-based Attendance
    path('attendance/mark/', views.mark_attendance_camera, name='mark_attendance_camera'),
    path('attendance/process/', views.process_attendance, name='process_attendance'),

    # Attendance List
    path('attendance/list/', views.emp_attendance_list, name='emp_attendance_list'),
    
    # Employee Management
    path('employees/', views.employee_list, name='employee_list'),
    path('employees/<int:pk>/', views.emp_detail, name='emp_detail'),
    path('employees/<int:pk>/authorize/', views.emp_authorize, name='emp_authorize'),
    path('employees/<int:pk>/delete/', views.emp_delete, name='emp_delete'),
    
    # Auth
    path('login/', views.user_login, name='login'),
    path('logout/', views.user_logout, name='logout'),
    
    # Camera Configuration
    path('cameras/', views.camera_config_list, name='camera_config_list'),
    path('cameras/create/', views.camera_config_create, name='camera_config_create'),
    path('cameras/<int:pk>/update/', views.camera_config_update, name='camera_config_update'),
    path('cameras/<int:pk>/delete/', views.camera_config_delete, name='camera_config_delete'),
    
    # Old OpenCV view (kept for backend processing if needed, but not user-facing)
    path('capture/', views.capture_and_recognize, name='capture_and_recognize'),
]
