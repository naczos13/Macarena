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
        self.mp_pose = mp.solutions.pose.PoseLandmark
        self.action_hand = self.mp_pose.LEFT_INDEX
        self.base_part = self.mp_pose.NOSE
        self.action_radius = 20

    def update(self):
        self.capture = cv2.VideoCapture(0)
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, self.raw_frame) = self.capture.read()
                self.find_human_pose_in_image()
            time.sleep(0.01)

    def show_frame(self):
        # as imput is raw_frame
        try:
            self.frame = self.raw_frame.copy()
        except AttributeError:
            pass

        # check the landmarks
        try:
            # self.pose_detection_results = self.raw_pose_detection_results.copy()
            self.landmarks = self.pose_detection_results.pose_landmarks.landmark
            self.draw_action_hand()
            self.draw_base_body_part()
            if self.done_what_simon_said():
                self.regenerate_simon_instruction()
        except (KeyError, AttributeError):
            pass

        # Display frames in main program
        cv2.imshow(self.PROGRAM_NAME, self.frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
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
                self.frame,
                self.pose_detection_results.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS,
            )
        except Exception:
            sys.exit("Error in draw_the_body_landmarks probably run before detection")

    def get_body_part_window_coordinate(self, body_index):
        body_part = self.landmarks[body_index.value]
        x = round(body_part.x * self.frame.shape[1])
        y = round(body_part.y * self.frame.shape[0])
        return x, y

    def draw_action_hand(self):
        x, y = self.get_body_part_window_coordinate(self.action_hand)
        cv2.circle(self.frame, (x, y), self.action_radius, (0, 255, 0), -1)

    def draw_base_body_part(self):
        x, y = self.get_body_part_window_coordinate(self.base_part)
        cv2.circle(self.frame, (x, y), self.action_radius, (255, 0, 0), -1)
        
    def done_what_simon_said(self):
        action_x, action_y = self.get_body_part_window_coordinate(self.action_hand)
        base_x, base_y = self.get_body_part_window_coordinate(self.base_part)
        
        distance = math.dist(
            [action_x, action_y],
            [base_x, base_y])
        
        return distance < self.action_radius
        
        
    def regenerate_simon_instruction(self):
        self.action_hand = random.choice([
            self.mp_pose.LEFT_INDEX,
            self.mp_pose.RIGHT_INDEX
        ])
        
        if self.action_hand == self.mp_pose.LEFT_INDEX:
            self.base_part = random.choice([
                self.mp_pose.NOSE,
                self.mp_pose.RIGHT_SHOULDER,
                self.mp_pose.RIGHT_INDEX,
                self.mp_pose.RIGHT_ELBOW
            ])
        elif self.action_hand == self.mp_pose.RIGHT_INDEX:
            self.base_part = random.choice([
                self.mp_pose.NOSE,
                self.mp_pose.LEFT_SHOULDER,
                self.mp_pose.LEFT_INDEX,
                self.mp_pose.LEFT_ELBOW
            ])
            
