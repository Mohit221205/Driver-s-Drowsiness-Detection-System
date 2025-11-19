import cv2
import time
import winsound  # For beeping sound on Windows

# Load the pre-trained face and eye detection models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')  # More accurate eye detection

# Open the default camera
cam = cv2.VideoCapture(0)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

# Track eye closure duration
eyes_closed_start_time = None
EYES_CLOSED_THRESHOLD = 1  # seconds

while True:
    ret, frame = cam.read()
    if not ret:
        break
    
    # Convert frame to grayscale for face and eye detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(50, 50))
    
    # Process each detected face
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=6, minSize=(20, 20))
        
        # Determine eye state based on detected eyes
        if len(eyes) >= 0.5:
            eye_state = "Eyes Open and Driver is Awake"
            color = (0, 255, 0)  # Green
            eyes_closed_start_time = None  # Reset closed eyes timer
        else:
            eye_state = "Eyes Closed and Driver is Drowsy state"
            color = (0, 0, 255)  # Red
            
            if eyes_closed_start_time is None:
                eyes_closed_start_time = time.time()
            else:
                elapsed_time = time.time() - eyes_closed_start_time
                if elapsed_time > EYES_CLOSED_THRESHOLD:
                    winsound.Beep(1000, 500)  # Beep sound (1000 Hz, 500 ms)
                    eyes_closed_start_time = None  # Reset after beep
        
        # Draw rectangles and display text only if a face is detected
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, eye_state, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0), 2)
    
    # Display the captured frame
    cv2.imshow('Camera', frame)
    
    # Press 's' to exit the loop
    if cv2.waitKey(1) == ord('s'):
        break

# Release the capture and writer objects
cam.release()
out.release()
cv2.destroyAllWindows()
