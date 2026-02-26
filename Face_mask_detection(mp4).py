import sys
# Importing the required packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os

def detect_and_predict_mask(frame, facenet, maskNet, confidence_threshold):
    (h,w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (300,300),(104.0,177.0,123.0)
    )
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > confidence_threshold:
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            (startX, startY, endX, endY) = box.astype("int")

            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w-1, endX)
            endY = min(h-1, endY)

            face = frame[startY:endY, startX:endX]

            # IF IT IS A EMPTY FACE,
            if face.size == 0:
                continue

            face =cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)
    return (locs, preds)

# Argument Parsing
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
    default="face_detector",
    help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
    default="mask_detector.model",
    help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Loading the face detector
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Loading the mask detector model
print("[INFO] Loading face msk detector model....")
maskNet = load_model(args["model"])

# Load test Image
print("[INFO] opening video file...")

cap = cv2.VideoCapture("Face_mask_video.mp4")  # your video file name

if not cap.isOpened():
    print("Error opening video file")
    sys.exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break  # End of video

    frame = cv2.resize(frame, (600, 600))

    (locs, preds) = detect_and_predict_mask(
        frame, faceNet, maskNet, args["confidence"]
    )

    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        label = "Mask" if mask > withoutMask else "No Mask"
        confidence = max(mask, withoutMask) * 100
        label_text = f"{label}: {confidence:.2f}%"

        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        cv2.putText(frame, label_text,
                    (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)

        cv2.rectangle(frame,
                      (startX, startY),
                      (endX, endY),
                      color, 2)

    cv2.imshow("Mask Detection - Video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
