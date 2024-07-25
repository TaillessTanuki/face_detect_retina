import cv2
import os
import time
from threading import Thread
import dlib # no need to install cmake
from retinaface import RetinaFace

# Global variables
faces = []
frameToProcess = None
isProcessingFrame = False
trackers = {}
next_id = 0
output_dir = 'captured_images'
os.makedirs(output_dir, exist_ok=True)


def overlap(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    return not (x1 > x2 + w2 or x1 + w1 < x2 or y1 > y2 + h2 or y1 + h1 < y2)

def faceProcessor():
    global faces
    global isProcessingFrame
    global frameToProcess
    global trackers
    global next_id

    while True:
        if isProcessingFrame == False:
            isProcessingFrame = True
            if frameToProcess is None:
                isProcessingFrame = False
                time.sleep(0.5)
                continue

            # Perform inference with RetinaFace
            detections = RetinaFace.detect_faces(frameToProcess)

            detected_faces = []
            if isinstance(detections, dict):
                # print(f'detections: {detections}')
                for face_id, face_info in detections.items():
                    print(f'face_id: {face_id}')
                    print(f'face_info: {face_info} \n')
                    x1, y1, x2, y2 = face_info['facial_area']
                    bbox = (x1, y1, x2 - x1, y2 - y1)
                    detected_faces.append(bbox)

            # Update trackers
            new_trackers = {}
            for bbox in detected_faces:
                matched = False
                # print(f'trackers: {trackers}')
                for tid, tracker in trackers.items():
                    tracker.update(frameToProcess)
                    pos = tracker.get_position()
                    # print(f'pos: {pos}')
                    tracked_bbox = (int(pos.left()), int(pos.top()), int(pos.width()), int(pos.height()))
                    # print(f'tracked_bbox: {tracked_bbox}')
                    # print(f'bbox: {bbox}')
                    # print(f'tracked_bbox: {tracked_bbox}')
                    if overlap(bbox, tracked_bbox):
                        new_trackers[tid] = tracker
                        # print(f'new_trackers: {new_trackers}')
                        matched = True
                        break
                if not matched:
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
                    tracker.start_track(frameToProcess, rect)
                    new_trackers[next_id] = tracker

                    # Save the image inside the bbox
                    x, y, w, h = bbox
                    face_img = frameToProcess[y: y + h, x: x + w]
                    cv2.imwrite(os.path.join(output_dir, f'face_{next_id}.jpg'), face_img)

                    next_id += 1
            trackers = new_trackers
            # print(f'trackers:  {trackers}')

            faces = [(tid, tracker.get_position()) for tid, tracker in trackers.items()]

            isProcessingFrame = False
        time.sleep(0.01)


def resize_with_aspect_ratio(frame, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = frame.shape[:2]

    if width is None and height is None:
        return frame

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(frame, dim, interpolation=inter)

def capture(source=None, width=None, height=None):
    global faces
    global isProcessingFrame
    global frameToProcess

    vid = cv2.VideoCapture(source)

    while True:
        ret, frame = vid.read()
        if not ret:
            print(f"Cannot receive frame from camera {source}. Exiting ...")
            break

        # Resize the frame while maintaining aspect ratio
        frame = resize_with_aspect_ratio(frame, width, height)

        if isProcessingFrame == False:
            frameToProcess = frame.copy()

        # Draw face bounding boxes with IDs
        for tid, pos in faces:
            x, y, w, h = int(pos.left()), int(pos.top()), int(pos.width()), int(pos.height())
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {tid}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow(f'Video {source}', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Start the capture thread
    c0 = Thread(target=capture, args=['rtsp://admin:cctv1234@192.168.88.2:554/11', 1280, 720], daemon=True).start()
    fp0 = Thread(target=faceProcessor, args=[], daemon=True).start()

    while True:
        time.sleep(1.0)
