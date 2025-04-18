from ultralytics import YOLO
import cv2
import math 
import threading
import socket
import logging
from logging.handlers import RotatingFileHandler
import time
# start webcam


HOST = '192.168.0.60'
PORTS = 2323
tags_logged = {}
tag_list = []
buffer_time = 5


# Logging Configuration
LOG_FILE = "server_logs.log"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log_handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3)
log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger = logging.getLogger(__name__)
logger.addHandler(log_handler)
logger.setLevel(logging.INFO)

# model
model = YOLO("yolo-Weights/yolov8n.pt")

# object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush","pen","truck","mouse"
              ]



def test_duplicate(tag_id, client_ip):
    global tags_logged, tag_list
    current_time_val = time.time()

    if tag_id not in tags_logged or (current_time_val - tags_logged[tag_id]["time"]) > buffer_time:
        tags_logged[tag_id] = {"time": current_time_val, "ip": client_ip}
        tag_list.append({"tag_id": tag_id, "ip": client_ip})  # Append tag with IP
        logger.info(f"Tag Detected: {tag_id} from {client_ip}")
    else:
        logger.info(f"Duplicate Tag Ignored: {tag_id} from {client_ip}")



def tag_reading():
    try:
        while True:
            data = client_socket.recv(1024)
            if not data:
                break  # Client disconnected

            # Decode received data
            encodedata = data.hex().upper()

            # Process tag ID if it starts with a specific prefix
            if encodedata.startswith("1100EE00"):
                tag_id = encodedata[8:32]
                print(tag_id)
                test_duplicate(tag_id, client_ip)
                logger.info(f"Data Received from {address}: {encodedata}")
    except Exception as e:
        print("Tag Reading Failed")
            





def start_server(port):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, port))
    server.listen(5)
    logger.info(f"Server started on {HOST}:{port}")

    try:
        client_socket, client_address = server.accept()
        logger.info(f"Connection accepted on port {port} from {client_address}")
        thread = threading.Thread(target=tag_reading, args=(client_socket, client_address)).start()
        thread2 = threading.Thread(target=Camera).start()
    except Exception as e:
        logger.error(f"Error on port {port}: {e}")
        




def Camera():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        success, img = cap.read()
        results = model(img, stream=True)

        # coordinates
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                # put box in cam
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # confidence
                confidence = math.ceil((box.conf[0]*100))/100
                print("Confidence --->",confidence)

                # class name
                cls = int(box.cls[0])
            

                if  classNames[cls] =="truck":
                    print("Truck Parked Start Reading Tag !!!! ")
                    break
                else:
                    pass    

                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()