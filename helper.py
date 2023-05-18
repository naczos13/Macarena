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
        ### Constans
        ## Tune mediapipe detection and tracking
        self.POSE_DETECTOR = mp.solutions.pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.PROGRAM_NAME = "Simon Says"
        self.FIRST_FRAME_PATH = "first_frame.jpg"
        self.ACTION_RADIUS = 20
        self.MP_POSE = mp.solutions.pose.PoseLandmark
        
        # Start the thread to read frames from the video stream
        self._thread = Thread(target=self.update, args=())
        self._thread.daemon = True
        self._thread.start()

        self._action_hand = self.MP_POSE.LEFT_INDEX # TODO change to null, wait for user to start
        self._base_part = self.MP_POSE.NOSE # TODO change to null, wait for user to start
        self._highest_score = 0
        self._current_score = 0
        self._capture = None # Camera capture
        self._landmarks = None # Body parts position landmarks
        self._left_menu = cv2.imread('menu_bar.jpg')    

    def update(self):
        self._capture = cv2.VideoCapture(0)
        # Read the next frame from the stream in a different thread
        while True:
            if self._capture.isOpened():
                (_, self._raw_frame) = self._capture.read()
                self._pose_detection_results = self.find_human_pose_in_image(raw_camera_frame=self._raw_frame)
            time.sleep(0.01)

    def show_frame(self):
        # input is raw_frame
        try:
            frame = self._raw_frame.copy()
        except AttributeError:
            pass

        # check the landmarks
        try:
            self._landmarks = self._pose_detection_results.pose_landmarks.landmark
            frame = self.draw_action_hand(camera_frame=frame, action_hand=self._action_hand, landmarks=self._landmarks)
            frame = self.draw_base_body_part(camera_frame=frame, target_body_part=self._base_part, landmarks=self._landmarks)
            if self.done_what_simon_said(active_hand=self._action_hand,
                                         target_body_part=self._base_part,
                                         landmarks=self._landmarks,
                                         camera_frame=frame):
                self._action_hand, self._base_part = self.regenerate_simon_instruction()
            #self.draw_scores()
        except (KeyError, AttributeError):
            pass

        # Display frames in main program
        canvas = self.concate_camera_frame_with_menu(camera_frame=frame, right_bar=self._left_menu)
        cv2.imshow(self.PROGRAM_NAME, canvas)
        key = cv2.waitKey(1)
        if key == ord("q"):
            self._capture.release()
            cv2.destroyAllWindows()
            exit(0)

    def show_menu(self):
        cv2.namedWindow(self.PROGRAM_NAME)
        # read to self._raw_frame because this picture need to be display until the camera frame is ready
        self._raw_frame = cv2.imread(self.FIRST_FRAME_PATH)
        if self._raw_frame is not None and self._left_menu is not None:
            canvas = self.concate_camera_frame_with_menu(camera_frame=self._raw_frame, right_bar=self._left_menu)
            cv2.imshow(self.PROGRAM_NAME, canvas)
        else:
            exit(1)

    def concate_camera_frame_with_menu(self, camera_frame, right_bar):
        window_width = right_bar.shape[1] + camera_frame.shape[1]
        canvas = np.zeros((max(right_bar.shape[0], camera_frame.shape[0]), window_width, 3), dtype=np.uint8)    
        # Place the camera frame on the left side of the canvas
        canvas[:camera_frame.shape[0], :camera_frame.shape[1]] = camera_frame
        # Place the menu bar on the right side of the canvas
        canvas[:right_bar.shape[0], camera_frame.shape[1]:] = right_bar
        return canvas
        
    def find_human_pose_in_image(self, raw_camera_frame):
        # Recolor image because mediapipe need RGB,
        # and cv2 has default BGR
        image = cv2.cvtColor(raw_camera_frame, cv2.COLOR_BGR2RGB)
        # Memory optimization
        image.flags.writeable = False

        # Make detection
        pose_detection_results = self.POSE_DETECTOR.process(image)
        return pose_detection_results

    def draw_the_body_landmarks(self):
        try:
            mp.solutions.drawing_utils.draw_landmarks(
                self.frame,
                self.pose_detection_results.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS,
            )
        except Exception:
            sys.exit("Error in draw_the_body_landmarks probably run before detection")

    def get_body_part_window_coordinate(self, body_index, landmarks, camera_frame):
        body_part = landmarks[body_index.value]
        x = round(body_part.x * camera_frame.shape[1])
        y = round(body_part.y * camera_frame.shape[0])
        return x, y

    def draw_action_hand(self, action_hand, camera_frame, landmarks):
        x, y = self.get_body_part_window_coordinate(body_index=action_hand, landmarks=landmarks, camera_frame=camera_frame)
        cv2.circle(camera_frame, (x, y), self.ACTION_RADIUS, (0, 255, 0), -1)
        return camera_frame

    def draw_base_body_part(self, target_body_part, camera_frame, landmarks):
        x, y = self.get_body_part_window_coordinate(body_index=target_body_part, landmarks=landmarks, camera_frame=camera_frame)
        cv2.circle(camera_frame, (x, y), self.ACTION_RADIUS, (255, 0, 0), -1)
        return camera_frame
        
    def done_what_simon_said(self, active_hand, target_body_part, landmarks, camera_frame):
        active_x, active_y = self.get_body_part_window_coordinate(body_index=active_hand, landmarks=landmarks, camera_frame=camera_frame)
        target_x, target_y = self.get_body_part_window_coordinate(body_index=target_body_part, landmarks=landmarks, camera_frame=camera_frame)
        
        distance = math.dist(
            [active_x, active_y],
            [target_x, target_y])
        
        return distance < self.ACTION_RADIUS
        
    def regenerate_simon_instruction(self):
        active_hand = random.choice([
            self.MP_POSE.LEFT_INDEX,
            self.MP_POSE.RIGHT_INDEX
        ])
        
        if active_hand == self.MP_POSE.LEFT_INDEX:
            target_body_part = random.choice([
                self.MP_POSE.NOSE,
                self.MP_POSE.RIGHT_SHOULDER,
                self.MP_POSE.RIGHT_INDEX,
                self.MP_POSE.RIGHT_ELBOW
            ])
        elif active_hand == self.MP_POSE.RIGHT_INDEX:
            target_body_part = random.choice([
                self.MP_POSE.NOSE,
                self.MP_POSE.LEFT_SHOULDER,
                self.MP_POSE.LEFT_INDEX,
                self.MP_POSE.LEFT_ELBOW
            ])
            
        return active_hand, target_body_part
            
        
            
