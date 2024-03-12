import cv2
import numpy as np
import onnxruntime
import torch
import matplotlib.pyplot as plt
import time
from transformers import DetrImageProcessor
import supervision as sv

# Define your DetrObjectDetectionOutput class and id2label mapping here
class DetrObjectDetectionOutput:
    def __init__(self, logits, pred_boxes):
        self.logits = logits
        self.pred_boxes = pred_boxes
    
    def __repr__(self):
        return f"DetrObjectDetectionOutput(logits={self.logits}, pred_boxes={self.pred_boxes})"

id2label = {
    1: "bicycle",
    2: "bus",
    3: "car",
    4: "motorbike",
    5: "person",
    6: "truck",
}

# Load the ONNX model
onnx_model_path = './weights/model.onnx'
ort_session = onnxruntime.InferenceSession(onnx_model_path, providers=[
                    (
                        "TensorrtExecutionProvider",
                        {
                                    'device_id': 0,

                            "trt_fp16_enable": True,
                            "trt_max_workspace_size": 2147483648,
                        },
                    ),
                    "CUDAExecutionProvider",
                ],)# Initialize the camera capture
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

# Create an instance of DetrImageProcessor (replace with your own implementation)
image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

# Function to preprocess the image
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = cv2.resize(image, (512, 512))  # Resize to match model input size
    image = image.astype(np.float32) / 255.0  # Normalize pixel values
    image = image.transpose(2, 0, 1)  # Channels-first format (C,H,W)
    return image

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Preprocess the frame
    image = preprocess_image(frame)

    # Run inference
    start = time.perf_counter()
    outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: np.expand_dims(image, axis=0)})
    print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
    # Convert outputs to torch tensors
    logits_tensor = torch.tensor(outputs[0])
    pred_boxes_tensor = torch.tensor(outputs[1])

    # Create an instance of DetrObjectDetectionOutput
    detr_output = DetrObjectDetectionOutput(logits_tensor, pred_boxes_tensor)
    
    # Determine the batch size
    batch_size = logits_tensor.shape[0]

    # Generate target sizes for the batch
    target_sizes = [[image.shape[1], image.shape[2]]] * batch_size
    target_sizes_tensor = torch.tensor(target_sizes)

    # Perform post-processing to get detections
    results = image_processor.post_process_object_detection(detr_output, target_sizes=target_sizes_tensor, threshold=0.2)[0]

    # Convert detections to Supervisely format
    detections = sv.Detections.from_transformers(transformers_results=results)

    # Get labels for detections
    labels = [f"{id2label[class_id.item()]}: {confidence:.2f}" for class_id, confidence in zip(detections.class_id, detections.confidence)]

    # Annotate the frame with detections and labels
    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

    # Display the annotated frame
    cv2.imshow('Object Detection', annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
