import cv2
import numpy as np
import onnxruntime as ort

# Load the ONNX model
model_path = "gesture_model.onnx"
ort_session = ort.InferenceSession(model_path)

# Preprocess the image for ONNX model input
def preprocess_frame(frame):
    # Resize to model input size (e.g., 224x224)
    frame_resized = cv2.resize(frame, (224, 224))
    # Convert to RGB if required by the model
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_RGBA2RGB)
    # Normalize pixel values to [0, 1]
    frame_normalized = frame_rgb.astype(np.float32) / 255.0
    # Transpose to (1, 3, 224, 224) format for PyTorch models
    frame_transposed = np.transpose(frame_normalized, (2, 0, 1))
    # Add batch dimension
    input_data = np.expand_dims(frame_transposed, axis=0)
    return input_data

# Postprocess the model output
def postprocess_output(output):
    # Get class with highest confidence
    class_index = np.argmax(output[0])
    # Map class index to gesture label
    gesture_labels = ["Thumbs Up", "Thumbs Down", "Open Palm", "Fist", "Victory", "No Gesture"]
    gesture = gesture_labels[class_index]
    confidence = output[0][class_index]
    return gesture, confidence

# Recognize gesture from an image
def recognize_gesture(frame):
    # Preprocess the frame
    input_data = preprocess_frame(frame)

    # Run inference
    ort_inputs = {ort_session.get_inputs()[0].name: input_data}
    ort_outs = ort_session.run(None, ort_inputs)

    # Postprocess the output
    gesture, confidence = postprocess_output(ort_outs)

    return f"Gesture: {gesture}, Confidence: {confidence:.2f}"

# Process video from camera or file
def process_video(source=0):
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Error: Cannot open video source.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Recognize gesture from the current frame
        result = recognize_gesture(frame)

        # Display result on the frame
        cv2.putText(frame, result, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the frame
        cv2.imshow("Gesture Recognition", frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Run gesture recognition on live camera feed
if __name__ == "__main__":
    process_video(0)  # 0 for webcam or provide video path
