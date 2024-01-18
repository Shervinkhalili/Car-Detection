import cv2

# Load the pre-trained car classifier (you may need to adjust the path)
car_cascade = cv2.CascadeClassifier('haarcascade_cars.xml')

# Load the video file
video_path = 'car1.mp4'
cap = cv2.VideoCapture(video_path)

# Get the video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create a video writer object to save the result
output_path = 'car_detection_output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform car detection with smaller scaleFactor
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(80, 80))

    # Adjust the rectangles to be higher
    for (x, y, w, h) in cars:
        # Calculate the center of the rectangle
        center_x = x + w // 2
        center_y = y + h // 2

        # Calculate the new height based on a scaling factor
        scale = 1.2  # Adjust this value to control the rectangle height increase
        new_height = int(h * scale)

        # Calculate the new top-left corner coordinates
        new_y = center_y - new_height // 2

        # Draw the adjusted rectangle
        cv2.rectangle(frame, (x, new_y), (x + w, new_y + new_height), (0, 255, 0), 2)

    # Write the frame with the detected objects to the output video
    output.write(frame)

    # Display the resulting frame
    cv2.imshow('Car Detection', frame)

    # Exit if 'x' is pressed
    if cv2.waitKey(25) & 0xFF == ord('x'):
        break

# Release the video capture object, video writer, and close the windows
cap.release()
output.release()
cv2.destroyAllWindows()
