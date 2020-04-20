import cv2
import numpy as np
    
confid, thresh = 0.5, 0.5

video_path = "./videos/video.mp4"
output_video_path = "./videos/output.mp4"
yolov3_weight_path = "./models/yolov3.weights"
yolov3_configuration_path = "./models/yolov3.cfg"
coco_label_path = "./models/coco.names"
np.random.seed(42)

model_network = cv2.dnn.readNetFromDarknet(yolov3_configuration_path, yolov3_weight_path)
network_layers = model_network.getLayerNames()
network_layers = [network_layers[i[0] - 1] for i in model_network.getUnconnectedOutLayers()]
LABELS = open(coco_label_path).read().strip().split("\n")

(frame_width, frame_height) = (None, None)
video_stream = cv2.VideoCapture(video_path)
writer = None

# Calibration needed for each video
def calibrated_dist(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + 550 / ((p1[1] + p2[1]) / 2) * (p1[1] - p2[1]) ** 2) ** 0.5

def isclose(p1, p2):
    c_d = calibrated_dist(p1, p2)
    calib = (p1[1] + p2[1]) / 2
    if 0 < c_d < 0.15 * calib:
        return 1
    elif 0 < c_d < 0.2 * calib:
        return 2
    else:
        return 0

while True:
    (grabbed, frame) = video_stream.read()

    if not grabbed:
        break

    (frame_height, frame_width) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    model_network.setInput(blob)
    layerOutputs = model_network.forward(network_layers)

    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if LABELS[classID] == "person":
                if confidence > confid:
                    box = detection[0:4] * np.array([frame_width, frame_height, frame_width, frame_height])
                    (centerX, centerY, width, height) = box.astype("int")
                    x, y = int(centerX - (width / 2)), int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confid, thresh)

    if len(idxs) > 0:
        status = list()
        idf = idxs.flatten()
        close_pair = list()
        s_close_pair = list()
        center = list()
        for i in idf:
            detected_object_dimension = boxes[i]
            (x, y) = (detected_object_dimension[0], detected_object_dimension[1])
            (w, h) = (detected_object_dimension[2], detected_object_dimension[3])
            center.append([int(x + w / 2), int(y + h / 2)])
            status.append(0)
        for i in range(len(center)):
            for j in range(len(center)):
                g = isclose(center[i], center[j])
                if g == 1:
                    close_pair.append([center[i], center[j]])
                    status[i], status[j] = 1, 1
                elif g == 2:
                    s_close_pair.append([center[i], center[j]])
                    if status[i] != 1:
                        status[i] = 2
                    if status[j] != 1:
                        status[j] = 2

        total_people = len(center)
        low_risk_people = status.count(2)
        high_risk_people = status.count(1)
        safe_people = status.count(0)

        # print(total_people, low_risk_people, high_risk_people, safe_people)

        contact_types = {
            'high': (0, 0, 150),
            'medium': (0, 120, 255),
            'low': (0, 255, 0)
        }

        count = 0
        for i in idf:
            risk_type = None
            detected_object_dimension = boxes[i]
            (x, y) = (detected_object_dimension[0], detected_object_dimension[1])
            (w, h) = (detected_object_dimension[2], detected_object_dimension[3])
            if status[count] == 1: 
                risk_type = 'high'
            elif status[count] == 0: 
                risk_type = 'low'
            else: 
                risk_type = 'medium'
            cv2.rectangle(frame, (x, y), (x + w, y + h), contact_types[risk_type], 2)
            count += 1

        for h in close_pair:
            cv2.line(frame, tuple(h[0]), tuple(h[1]), (0, 0, 255), 2)
        for b in s_close_pair:
            cv2.line(frame, tuple(b[0]), tuple(b[1]), (0, 255, 255), 2)

        cv2.imshow('Social distancing analyser', frame)
        cv2.waitKey(1)

    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(output_video_path, fourcc, 30, (frame.shape[1], frame.shape[0]), True)

    writer.write(frame)
print("Processing finished: open output.mp4")
writer.release()
video_stream.release()