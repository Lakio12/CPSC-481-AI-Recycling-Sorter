#Be sure to have opencv-python installed
# Handles webcam access and image capture.
# Implemented currently for picture to overwrite the previous one for now 

import cv2


def capture_photo():
    # Initialize the default webcam
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Check if the camera is available
    if not camera.isOpened():
        camera.release()
        print("Error: Unable to access camera.")
        return None

    try:
        while True:
            # Read the current camera frame
            success, frame = camera.read()

            if not success:
                print("Error: Unable to read camera.")
                return None

            window_name = "ML Recycling Sorter - Camera"

            # Display live camera preview
            cv2.imshow(window_name, frame)

            # Wait for keyboard input
            key = cv2.waitKey(1) & 0xFF

            # Press SPACE to capture image
            if key == 32:
                image_path = "captured_image.jpg"

                if cv2.imwrite(image_path, frame):
                    print("Image captured successfully!")
                    return image_path

                return None

            # Check if user clicked the X button
            if cv2.getWindowProperty(
                window_name, cv2.WND_PROP_VISIBLE
                ) < 1:
                print("Camera window closed.")
                return None

            # Press Q or ESC to cancel
            elif key in (ord("q"), 27):
                return None

    finally:
        # Close the camera
        camera.release()
        cv2.destroyAllWindows()


# If testing file independently
if __name__ == "__main__":
    capture_photo()
