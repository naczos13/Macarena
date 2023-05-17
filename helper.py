import cv2
import mediapipe as mp
import numpy as np
import random
import math
from threading import Thread
import time
import sys


class VideoStreamWidget(object):
    def __init__(self):
        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        self.PROGRAM_NAME = "Simon Says"
        self.FIRST_FRAME_PATH = "first_frame.jpg"
        ## Tune mediapipe detection and tracking
        self.pose_detector = mp.solutions.pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.action_hand = 'LEFT_INDEX'
        self.mp_pose = mp.solutions.pose.PoseLandmark

    def update(self):
        self.capture = cv2.VideoCapture(0)
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, self.raw_frame) = self.capture.read()
                self.find_human_pose_in_image()
            time.sleep(.01)
    
    def show_frame(self):
        # as imput is raw_frame
        try:
            self.frame = self.raw_frame.copy()
        except AttributeError:
            pass
        
        # check the landmarks
        try:
            #self.pose_detection_results = self.raw_pose_detection_results.copy()
            self.landmarks = self.pose_detection_results.pose_landmarks.landmark
            self.draw_action_hand()
        except (KeyError, AttributeError):
            pass
        
        # Display frames in main program
        cv2.imshow(self.PROGRAM_NAME, self.frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(0)
    
    def show_menu(self):    
        cv2.namedWindow(self.PROGRAM_NAME)
        self.frame = cv2.imread(self.FIRST_FRAME_PATH)
        if self.frame is not None:
            cv2.imshow(self.PROGRAM_NAME, self.frame)
        else:
            exit(1)
    
            
    def find_human_pose_in_image(self):
        # Recolor image because mediapipe need RGB,
        # and cv2 has default BGR
        image = cv2.cvtColor(self.raw_frame, cv2.COLOR_BGR2RGB)
        # Memory optimization
        image.flags.writeable = False

        # Make detection
        self.pose_detection_results = self.pose_detector.process(image)
        
        # # Memory optimization
        # self.frame.flags.writeable = True
        # # Recolor image back
        # self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)


    def draw_the_body_landmarks(self):
        try:
            mp.solutions.drawing_utils.draw_landmarks(
                    self.frame, self.pose_detection_results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS
                )
        except Exception:
            sys.exit("Error in draw_the_body_landmarks probably run before detection")


    def get_body_part_window_coordinate(self, body_index):
        body_part = self.landmarks[body_index]
        x = round(body_part.x * self.frame.shape[1])
        y = round(body_part.y * self.frame.shape[0])
        return x, y


    def draw_action_hand(self):
        if self.action_hand == self.mp_pose.LEFT_INDEX.name:
            action_x, action_y = self.get_body_part_window_coordinate(self.mp_pose.LEFT_INDEX.value)
        elif self.action_hand == self.mp_pose.RIGHT_INDEX.name:
            action_x, action_y = self.get_body_part_window_coordinate(self.mp_pose.RIGHT_INDEX.value)
        else:
            return None

        cv2.circle(self.frame, (action_x, action_y), 20, (0, 255, 0), -1)


mouse_clicked = False
action_part = ""
base_part = ""
simon_message = ""

def mouse_click_event(event, x, y, flags, param):
    global mouse_clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        if x < 100 and y < 100:
            mouse_clicked = True

def update_layout(frame, landmarks):
    global mouse_clicked
    global action_part
    global base_part
    global simon_message
    if mouse_clicked:
        action_part, base_part, simon_message = regenerate_simon_instruction()
        mouse_clicked = False
        
    draw_action_hand(frame, landmarks, action_part)
    draw_base_body_part(frame, landmarks, base_part)
    
    print_what_right_hand_touch(landmarks, frame)
    
    disable_head_landmarks(landmarks)
    display_arm_angles(frame, landmarks)
        
    generate_text_with_instructions(frame, simon_message)
    genereate_text_with_score(frame, "SCORE: 25")
    generate_text_with_helper(frame, "Helper: is not ok")

def find_human_pose_in_image(frame):
    ## Tune mediapipe detection and tracking
    pose_detector = mp.solutions.pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    # Recolor image because mediapipe need RGB,
    # and cv2 has default BGR
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Memory optimization
    image.flags.writeable = False

    # Make detection
    results = pose_detector.process(frame)

    # Memory optimization
    image.flags.writeable = True
    # Recolor image back
    image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    return results

def display_arm_angles(image, landmarks):
    mp_pose = mp.solutions.pose
    # Get coordinates
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]

    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

    # Calculate angle
    left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

    # Vizualize left angle
    cv2.putText(
        image,
        str(left_angle),
        tuple(np.multiply([left_elbow.x, left_elbow.y], [640, 480]).astype(int)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Vizualize lright angle
    cv2.putText(
        image,
        str(right_angle),
        tuple(np.multiply([right_elbow.x, right_elbow.y], [640, 480]).astype(int)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def disable_head_landmarks(landmarks):
    # Clear face landmarks
    CUTOFF_THRESHOLD = 10  # head and face according to https://camo.githubusercontent.com/7fbec98ddbc1dc4186852d1c29487efd7b1eb820c8b6ef34e113fcde40746be2/68747470733a2f2f6d65646961706970652e6465762f696d616765732f6d6f62696c652f706f73655f747261636b696e675f66756c6c5f626f64795f6c616e646d61726b732e706e67
    for landmark_id, landmark in enumerate(landmarks):
        if landmark_id == 0:
            continue
        if landmark_id <= CUTOFF_THRESHOLD:
            landmark.visibility = 0


def regenerate_simon_instruction():
    action_parts = [
        { 'pose_name' : 'LEFT_INDEX', 'display_name' : 'LEFT HAND' },
        { 'pose_name' : 'RIGHT_INDEX', 'display_name' : 'RIGHT HAND' }
        ]
    base_parts = [
        { 'pose_name' : 'LEFT_SHOULDER', 'display_name' : 'LEFT SHOULDER' },
        { 'pose_name' : 'RIGHT_SHOULDER', 'display_name' : 'RIGHT SHOULDER' },
        { 'pose_name' : 'NOSE', 'display_name' : 'NOSE' }
        ]
    actions_to_do = ["on"]

    action_part = random.choice(action_parts)
    base_part = random.choice(base_parts)
    action_to_do = random.choice(actions_to_do)

    simon_says = "Simon Says!\n " + action_part['display_name'] + " " + action_to_do + " " + base_part['display_name']
    print(simon_says)
    return action_part['pose_name'], base_part['pose_name'], simon_says

def print_what_right_hand_touch(landmarks, image):
    mp_pose = mp.solutions.pose
    right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value]
    
    closes_body_part = {'name' : 'empty', 'distance' : 10, 'position_x' : 0, 'position_y' : 0}
    for single_pose in [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.NOSE]:     
        body_part = landmarks[single_pose.value]
        
        # compute distance
        distance = math.dist(
            [body_part.x, body_part.y],
            [right_hand.x, right_hand.y])
        
        if closes_body_part["distance"] > distance:
            closes_body_part["distance"] = distance
            closes_body_part["name"] = single_pose.name
            closes_body_part["position_x"] = body_part.x
            closes_body_part["position_y"] = body_part.y
         
    if closes_body_part["distance"] < 0.1:
        print(f"{mp_pose.PoseLandmark.RIGHT_INDEX.name}:({round(right_hand.x, 2)}, {round(right_hand.y, 2)}) is near {closes_body_part['name']}:({round(closes_body_part['position_x'], 2)}, {round(closes_body_part['position_y'], 2)}) and distance is {closes_body_part['distance']}") 
    
    #TODO
   
def get_body_part_window_coordinate(image, landmarks, body_index):
    body_part = landmarks[body_index]
    x = round(body_part.x * image.shape[1])
    y = round(body_part.y * image.shape[0])
    return x, y   
  
    
def draw_action_hand(image, landmarks, action_part):
    mp_pose = mp.solutions.pose

    if action_part == mp_pose.PoseLandmark.LEFT_INDEX.name:
        action_x, action_y = get_body_part_window_coordinate(image, landmarks, mp_pose.PoseLandmark.LEFT_INDEX.value)
    elif action_part == mp_pose.PoseLandmark.RIGHT_INDEX.name:
        action_x, action_y = get_body_part_window_coordinate(image, landmarks, mp_pose.PoseLandmark.RIGHT_INDEX.value)
    else:
        return None

    cv2.circle(image, (action_x, action_y), 20, (0, 255, 0), -1)
    
def draw_base_body_part(image, landmarks, base_part):
    mp_pose = mp.solutions.pose

    if base_part == mp_pose.PoseLandmark.LEFT_SHOULDER.name:
        base_x, base_y = get_body_part_window_coordinate(image, landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER.value)
    elif base_part == mp_pose.PoseLandmark.RIGHT_SHOULDER.name:
        base_x, base_y = get_body_part_window_coordinate(image, landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
    elif base_part == mp_pose.PoseLandmark.NOSE.name:
        base_x, base_y = get_body_part_window_coordinate(image, landmarks, mp_pose.PoseLandmark.NOSE.value)
    else:
        return None

    cv2.circle(image, (base_x, base_y), 20, (255, 0, 0), -1)


def calculate_angle(a, b, c):
    radians = np.arctan2(c.y - b.y, c.x - b.x) - np.arctan2(a.y - b.y, a.x - b.x)
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return round(angle)


def generate_text_with_instructions(image, text):
    # Set the font properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)  # White color

    # Get the size of the text
    text_size, _ = cv2.getTextSize(text, font, font_scale, 1)

    # Bottom middle
    text_x = (image.shape[1] - text_size[0]) // 2
    text_y = image.shape[0] - 20

    # Put the text on the image
    cv2.putText(image, text, (text_x, text_y), font, font_scale, font_color, 2)


def genereate_text_with_score(image, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)  # White color

    # Get the size of the text
    text_size, _ = cv2.getTextSize(text, font, font_scale, 1)

    # Right Top corner
    text_x = image.shape[1] - text_size[0] - 20
    text_y = 50

    # Put the text on the image
    cv2.putText(image, text, (text_x, text_y), font, font_scale, font_color, 2)


def generate_text_with_helper(image, text):
    # TODO change the color dependig if the user is in proper position
    font_color = (0, 0, 255)  # RED color

    # Left Top Corner
    text_x = 20
    text_y = 50

    # Put the text on the image
    cv2.putText(
        image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, font_color, 2
    )
