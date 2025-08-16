import cv2
import mediapipe as mp
import numpy as np

# Load Model Age & Gender 
face_proto = 'model/deploy.prototxt'
face_model = 'model/res10_300x300_ssd_iter_140000.caffemodel'
age_proto = 'model/age_deploy.prototxt'
age_model = 'model/age_net.caffemodel'
gender_proto = 'model/gender_deploy.prototxt'
gender_model = 'model/gender_net.caffemodel'

face_net = cv2.dnn.readNet(face_model, face_proto)
age_net = cv2.dnn.readNet(age_model, age_proto)
gender_net = cv2.dnn.readNet(gender_model, gender_proto)

age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)',
            '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

# Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
my_drawing_specs = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
mp_face_mesh = mp.solutions.face_mesh

def process_frame(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 [104, 117, 123], False, False)
    face_net.setInput(blob)
    detections = face_net.forward()

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype(int)
                face_img = frame[y1:y2, x1:x2].copy()
                if face_img.size == 0:
                    continue

                # Gender
                blob2 = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                              [104, 117, 123], swapRB=False)
                gender_net.setInput(blob2)
                gender_preds = gender_net.forward()
                gender = gender_list[gender_preds[0].argmax()]

                # Age
                age_net.setInput(blob2)
                age_preds = age_net.forward()
                age = age_list[age_preds[0].argmax()]

                label = f"{gender}, {age}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

                # Landmark
                face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(face_rgb)
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            image=face_img,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                        )
                        mp_drawing.draw_landmarks(
                            image=face_img,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=my_drawing_specs
                        )
                frame[y1:y2, x1:x2] = face_img
    return frame

def detect_from_frame(frame):
    return process_frame(frame)

def detect_from_image(img):
    img = cv2.flip(img, 1)
    return process_frame(img)
