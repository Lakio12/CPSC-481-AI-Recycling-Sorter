import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import queue
from PIL import Image, ImageTk, ImageOps
from App.camera import capture_photo

# Adjustable UI warning threshold, not a validated accuracy cutoff.
CONFIDENCE_WARNING_THRESHOLD = 0.60

DISPOSAL_GUIDANCE = {
    "cardboard": (
        "Recycle if accepted locally. Keep cardboard clean and dry "
        "and flatten boxes."
    ),
    "glass": (
        "Recycle bottles and jars if accepted locally. Empty and rinse them. "
        "Other glass items may require different disposal."
    ),
    "metal": (
        "Recycle food and drink cans if accepted locally; empty and rinse them. "
        "Other metal items may need a drop-off location."
    ),
    "paper": (
        "Recycle clean, dry paper if accepted locally."
    ),
    "plastic": (
        "Check whether your local program accepts this specific plastic item. "
        "Empty and rinse accepted containers. Do not assume bags or wrappers "
        "belong in curbside recycling."
    ),
    "trash": (
        "Use the trash bin only if this is ordinary garbage that cannot be "
        "recycled or composted locally. Batteries, electronics, and hazardous "
        "items require separate disposal."
    )
}

def get_disposal_guidance(category, confidence):
    guidance = DISPOSAL_GUIDANCE.get(
        category.lower(),
        "Check your local disposal instructions for this item."
    )

    if confidence < CONFIDENCE_WARNING_THRESHOLD:
        return (
            "Prediction uncertain: Verify the material before sorting.\n"
            "If the predicted category is correct: " + guidance
        )

    return guidance


def create_interface(root, controller = None):
    # Allow enough room for the disposal guidance.
    root.geometry("900x720")
    root.minsize(700, 680)

    state = {"image_path": None}
    prediction_queue = queue.Queue()
    model_cache = {}

    # Application title
    title = tk.Label(
        root,
        text = "ML Recycling Sorter",
        font = ("Arial", 24, "bold")
    )
    title.pack(pady=(15, 5))

    instructions = tk.Label(
        root,
        text = "Upload an image or take a picture of an item.",
        font = ("Arial", 11)
    )
    instructions.pack(pady = (0, 10))

    # Image preview
    preview_frame = tk.Frame(
        root,
        width = 400,
        height = 250,
        bg = "#D9D9D9",
        highlightbackground = "#888888",
        highlightthickness = 2
    )
    preview_frame.pack(pady=5)
    preview_frame.pack_propagate(False)

    image_label = tk.Label(
        preview_frame,
        text = "No image selected",
        bg = "#D9D9D9",
        font = ("Arial", 14)
    )
    image_label.pack(expand=True)

    status_text = tk.StringVar(
        value = "Choose an image to get started."
    )
    category_text = tk.StringVar(value="Category: —")
    confidence_text = tk.StringVar(value="Confidence: —")
    guidance_text = tk.StringVar(value="Disposal guidance: —")

    def reset_results():
        """Clear the previous prediction and guidance."""
        category_text.set("Category: —")
        confidence_text.set("Confidence: —")
        guidance_text.set("Disposal guidance: —")

    def display_image(image_path):
        """Open an image and update the preview."""
        try:
            with Image.open(image_path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.thumbnail((380, 230))
                photo = ImageTk.PhotoImage(image)
        except (OSError, ValueError):
            messagebox.showerror(
                "Unable to open image",
                "Please select a valid JPG, PNG, BMP, or WEBP image.",
                parent = root
            )
            return

        state["image_path"] = image_path
        image_label.config(image = photo, text = "")
        image_label.image = photo

        reset_results()
        status_text.set(
            "Image ready. Click Classify to identify the item."
        )
        classify_button.config(state = tk.NORMAL)

    def upload_image():
        """Choose an image from the computer."""
        image_path = filedialog.askopenfilename(
            parent = root,
            title = "Select an image",
            filetypes = [
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*")
            ]
        )

        if image_path:
            display_image(image_path)

    def take_picture():
        """Capture a photo using the existing webcam code."""
        try:
            image_path = capture_photo()
        except Exception as error:
            messagebox.showerror(
                "Camera Error",
                str(error),
                parent = root
            )
            return

        if image_path:
            display_image(image_path)

    def clear_image():
        """Clear the selected image and results."""
        state["image_path"] = None
        image_label.config(image = "", text = "No image selected")
        image_label.image = None

        reset_results()
        status_text.set("Choose an image to get started.")
        classify_button.config(state = tk.DISABLED)

    def prediction_worker(image_path):
        """Load the model and predict without blocking the UI."""
        try:
            from src.predict import load_model, predict_image

            # Load the model once and reuse it.
            if not model_cache:
                model, classes, device = load_model()
                model_cache.update(
                    model = model,
                    classes = classes,
                    device = device
                )

            # Match the orientation shown in the preview.
            with Image.open(image_path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")

            category, confidence = predict_image(
                image,
                model_cache["model"],
                model_cache["classes"],
                model_cache["device"]
            )

            prediction_queue.put(("success", category, confidence))

        except Exception as error:
            prediction_queue.put(("error", str(error)))

    def check_prediction():
        """Update the interface when the worker finishes."""
        try:
            result = prediction_queue.get_nowait()
        except queue.Empty:
            root.after(100, check_prediction)
            return

        for button in button_frame.winfo_children():
            button.config(state = tk.NORMAL)

        if result[0] == "success":
            category = result[1]
            confidence = result[2]

            category_text.set(f"Category: {category.capitalize()}")
            confidence_text.set(f"Confidence: {confidence:.1%}")
            guidance_text.set(
                "Disposal guidance: "
                + get_disposal_guidance(category, confidence)
            )
            status_text.set("Classification complete.")
        else:
            status_text.set("Classification failed. Please try again.")
            messagebox.showerror(
                "Classification Error",
                result[1],
                parent = root
            )

    def classify_selected_image():
        """Start classification of the selected image."""
        image_path = state["image_path"]

        if not image_path:
            messagebox.showinfo(
                "Select an image",
                "Upload an image or take a picture first.",
                parent = root
            )
            return

        reset_results()
        status_text.set(
            "Classifying..."
        )

        # Prevent image changes while classification is running.
        for button in button_frame.winfo_children():
            button.config(state = tk.DISABLED)

        threading.Thread(
            target = prediction_worker,
            args = (image_path,),
            daemon = True
        ).start()

        root.after(100, check_prediction)

    # Image controls
    button_frame = tk.Frame(root)
    button_frame.pack(pady = 10)

    tk.Button(
        button_frame,
        text = "Upload Image",
        command = upload_image,
        font = ("Arial", 11),
        padx = 12,
        pady = 6
    ).pack(side=tk.LEFT, padx=5)

    tk.Button(
        button_frame,
        text = "Take Picture",
        command=take_picture,
        font = ("Arial", 11),
        padx = 12,
        pady = 6
    ).pack(side = tk.LEFT, padx = 5)

    tk.Button(
        button_frame,
        text = "Clear",
        command = clear_image,
        font = ("Arial", 11),
        padx = 12,
        pady = 6
    ).pack(side = tk.LEFT, padx = 5)

    classify_button = tk.Button(
        button_frame,
        text = "Classify",
        command = classify_selected_image,
        state = tk.DISABLED,
        font = ("Arial", 11, "bold"),
        bg = "#228B22",
        fg = "white",
        padx = 12,
        pady = 6
    )
    classify_button.pack(side=tk.LEFT, padx=5)

    # Results area
    results_frame = tk.LabelFrame(
        root,
        text = "Classification Results",
        font = ("Arial", 12, "bold"),
        padx = 15,
        pady = 8
    )
    results_frame.pack(fill = tk.X, padx = 40, pady = 5)

    tk.Label(
        results_frame,
        textvariable = category_text,
        font = ("Arial", 12)
    ).pack(anchor = "w")

    tk.Label(
        results_frame,
        textvariable = confidence_text,
        font = ("Arial", 12)
    ).pack(anchor = "w", pady = (5, 0))

    guidance_label = tk.Label(
        results_frame,
        textvariable = guidance_text,
        font = ("Arial", 11),
        justify = tk.LEFT,
        anchor = "w",
        wraplength = 700
    )
    guidance_label.pack(fill=tk.X, pady=(10, 0))

    def resize_guidance(event):
        """Wrap the guidance to fit the results area."""
        guidance_label.config(
            wraplength = max(200, event.width - 40)
        )

    results_frame.bind("<Configure>", resize_guidance)

    tk.Label(
        root,
        textvariable = status_text,
        font = ("Arial", 10),
        wraplength = 600
    ).pack(pady = 8)
