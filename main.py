import cv2
import numpy as np
import pyttsx3

language = 'en'

net = cv2.dnn.readNet("yolo/custom-yolov4-tiny-detector_best.weights", "yolo/custom-yolov4-tiny-detector.cfg")

classes = []
with open("yolo/coco.names", "r") as f:
    classes = f.read().splitlines()

cap = cv2.VideoCapture("yolo/Viper Snake in Sri lanka - ලන්කවෙ ඉන්න තිත් පොලගා.mp4")

# img = cv2.imread("yolo/pic.jpg")
while True:
    _, img = cap.read()
    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

    net.setInput(blob)

    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    class_ids = []
    boxes = []
    confidences = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)

                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))

    for i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i], 2))
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label + " " + confidence, (x, y + 20), font, 2, (255, 0, 0), 2)
        engine = pyttsx3.init()
        voices = engine.getProperty("voices")
        engine.setProperty("voices", voices[1].id)
        engine.setProperty("rate", 150)
        engine.say(label + "detected")
        engine.runAndWait()

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
