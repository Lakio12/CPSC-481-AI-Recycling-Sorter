import tkinter as tk
from PIL import Image, ImageTk
from App.camera import capture_photo
#from App.styles import "stuff"


def create_interface(root, controller=None):

    # Application title
    title = tk.Label(
        root,
        text="ML Recycling Sorter",
        font=("Arial", 24, "bold")
    )
    title.pack(pady=20)

    # Image preview container
    preview_frame = tk.Frame(
        root,
        width=400,
        height=300,
        bg="#D9D9D9",
        highlightbackground="#888888",
        highlightthickness=2
    )
    preview_frame.pack(pady=20)
    preview_frame.pack_propagate(False)

    # Image preview label
    image_label = tk.Label(
        preview_frame,
        text="No image selected",
        bg="#D9D9D9",
        font=("Arial", 14)
    )
    image_label.pack(expand=True)

    # Display the captured image
    def display_image(image_path):
        image = Image.open(image_path)
        image.thumbnail((380, 280))

        photo = ImageTk.PhotoImage(image)

        image_label.config(image=photo, text="")
        image_label.image = photo

    # Capture a photo using the webcam
    def take_picture():
        image_path = capture_photo()

        if image_path:
            display_image(image_path)

    # Camera button
    camera_button = tk.Button(
        root,
        text="Take Picture",
        command=take_picture,
        font=("Arial", 12),
        padx=15,
        pady=8
    )
    camera_button.pack(pady=10)
