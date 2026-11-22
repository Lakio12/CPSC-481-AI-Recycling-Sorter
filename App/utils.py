from src.predict import predict_image
# from src.disposal import get_disposal_guidance


def validate_image(image_path):
    # Validate the selected image
    # Implementation to be added
    pass


def classify_image(image_path):
    # Request a prediction from the AI model
    prediction = predict_image(image_path)

    return prediction


def process_image(image_path):
    # Validate the image
    if not validate_image(image_path):
        return None

    # Classify the image
    prediction = classify_image(image_path)

    # Retrieve disposal guidance (Not implemented yet)
    # guidance = get_disposal_guidance(
    #     prediction["category"]
    # )

    # Return results to the interface
    return {
        "prediction": prediction,
        # "guidance": guidance
    }
