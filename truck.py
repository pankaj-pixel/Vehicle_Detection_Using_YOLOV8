from ultralytics import YOLO
import cv2
import math
import threading
import socket
import logging
from logging.handlers import RotatingFileHandler
import time
from datetime import datetime

# === CONFIGURATION ===
HOST = '192.168.0.73'   # Your local server IP
PORT = 2323             # Tag reader port
BUFFER_TIME = 5         # Time buffer for duplicate tag reads

# === Logging Configuration ===
LOG_FILE = "server_logs.log"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log_handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3)
log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger = logging.getLogger(__name__)
logger.addHandler(log_handler)

# === Model Setup ===
model = YOLO("yolo-Weights/yolov8n.pt") 
TARGET_CLASS = "truck"

# === Globals ===
tags_logged = {}
tag_list = []
truck_detected = False
read_tags = False
client_socket = None
client_ip = None
truck_start_time = None

# === Classes List ===
classNames = model.names

# === Duplicate Check ===
def test_duplicate(tag_id, client_ip):
    current_time = time.time()
    if tag_id not in tags_logged or (current_time - tags_logged[tag_id]["time"]) > BUFFER_TIME:
        tags_logged[tag_id] = {"time": current_time, "ip": client_ip}
        tag_list.append({"tag_id": tag_id, "ip": client_ip})
        logger.info(f"Tag Detected: {tag_id} from {client_ip}")
    else:
        logger.info(f"Duplicate Tag Ignored: {tag_id} from {client_ip}")

# === RFID Tag Reading Thread ===
def tag_reading():
    global client_socket, client_ip, read_tags
    try:
        while True:
            if not read_tags or client_socket is None:
                time.sleep(1)
                continue

            data = client_socket.recv(1024)
            if not data:
                break

            encoded = data.hex().upper()
            if encoded.startswith("1100EE00"):
                tag_id = encoded[8:32]
                print(f"[TAG READ] {tag_id}")
                test_duplicate(tag_id, client_ip)
    except Exception as e:
        print("Tag Reading Failed:", e)
        logger.error(f"Tag Reader Error: {e}")

# === Camera and YOLO Truck Detection ===
def camera_detection():
    global truck_detected, read_tags, truck_sqart_time

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        success, img = cap.read()
        if not success:
            continue

        results = model(img, stream=True)
        truck_present = False

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                confidence = round(float(box.conf[0]), 2)
                label = classNames[cls]

                if label == TARGET_CLASS and confidence > 0.5:
                    truck_present = True
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(img, f"{label} {confidence}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2)
                    break  # Only need one truck detection

        if truck_present and not truck_detected:
            truck_detected = True
            truck_start_time = datetime.now()
            read_tags = True
            logger.info(f"Truck Detected - Start Time: {truck_start_time}")
            print("Truck Detected. Started Tag Reading...")

        elif not truck_present and truck_detected:
            truck_detected = False
            read_tags = False
            truck_end_time = datetime.now()
            logger.info(f"Truck Left - End Time: {truck_end_time}")
            logger.info(f"Tags Read: {[tag['tag_id'] for tag in tag_list]}")
            tag_list.clear()

        cv2.imshow("YOLO Truck Detection", img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# === TCP Server Setup ===
def start_server():
    global client_socket, client_ip
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(1)
    print(f"Waiting for tag reader connection on {HOST}:{PORT}")
    logger.info(f"Waiting for tag reader connection on {HOST}:{PORT}")

    client_socket, client_address = server.accept()
    client_ip = client_address[0]
    print(f"Tag Reader Connected from {client_ip}")
    logger.info(f"Tag Reader Connected from {client_ip}")

    # Start tag reading thread
    threading.Thread(target=tag_reading, daemon=True).start()

# === MAIN ENTRY ===
if __name__ == "__main__":
    # Start server and camera in separate threads
    server_thread = threading.Thread(target=start_server)
    camera_thread = threading.Thread(target=camera_detection)

    server_thread.start()
    camera_thread.start()

    server_thread.join()
    camera_thread.join()
