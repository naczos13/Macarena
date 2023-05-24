import cv2
import mediapipe as mp
import numpy as np
import random
import math
from threading import Thread
import time
import json
from google.protobuf.json_format import MessageToDict


class VideoStreamWidget(object):
    def __init__(self, capture_input_from_camera=True):
        ### Constant
        ## Tune mediapipe detection and tracking
        self.POSE_DETECTOR = mp.solutions.pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.PROGRAM_NAME = "Simon Says"
        self.FIRST_FRAME_PATH = "static/first_frame.jpg"
        self.ACTION_RADIUS = 20
        self.CAMERA_FRAME_WIDTH = 640
        self.CAMERA_FRAME_HEIGHT = 480
        self.MENU_FRAME_WIDTH = 260
        self.MENU_FRAME_HEIGHT = 480
        self.MP_POSE = mp.solutions.pose.PoseLandmark
        self.PATH_TO_HIGH_SCORE = "static/high_score.txt"
        self.TIME_LIMIT_PER_ROUND = 30
        self.PATH_TO_SAVED_POSES = "static/saved_poses.json"
        self.HANDS = [self.MP_POSE.LEFT_INDEX, self.MP_POSE.RIGHT_INDEX]
        self.REACHABLE_BY_LEFT_HAND = [
            self.MP_POSE.RIGHT_SHOULDER,
            self.MP_POSE.RIGHT_INDEX,
            self.MP_POSE.RIGHT_ELBOW,
        ]
        self.REACHABLE_BY_RIGHT_HAND = [
            self.MP_POSE.LEFT_SHOULDER,
            self.MP_POSE.LEFT_INDEX,
            self.MP_POSE.LEFT_ELBOW,
        ]
        self.REACHABLE_BY_BOTH_HANDS = [
            self.MP_POSE.NOSE,
            # self.MP_POSE.LEFT_HIP,
            # self.MP_POSE.RIGHT_HIP,
            # self.MP_POSE.LEFT_KNEE,
            # self.MP_POSE.RIGHT_KNEE,
            # self.MP_POSE.LEFT_ANKLE,
            # self.MP_POSE.RIGHT_ANKLE,
        ]

        cv2.namedWindow(self.PROGRAM_NAME)
        cv2.setMouseCallback(self.PROGRAM_NAME, mouse_func)
        if capture_input_from_camera:
            # Start the thread to read frames from the video stream
            self._thread = Thread(target=self.update, args=())
            self._thread.daemon = True
            self._thread.start()

        self._action_hand, self._base_part = self.regenerate_simon_instruction()
        self._highest_score = self.read_highest_score_from_file(self.PATH_TO_HIGH_SCORE)
        self._current_score = 0
        self._capture = None  # Camera capture
        self._landmarks = None  # Body parts position landmarks
        self._raw_right_menu = cv2.imread("static/menu_bar.jpg")
        self._macarena_runs = False
        self._copy_pose_runs = False
        self._snapshot = False
        self._pose_id_read = 0
        self._need_restart_macarena = False
        self.show_welcome()

    def update(self):
        self._capture = cv2.VideoCapture(0)
        # Read the next frame from the stream in a different thread
        while True:
            if self._capture.isOpened():
                (_, self._raw_frame) = self._capture.read()
                self._pose_detection_results = self.find_human_pose_in_image(
                    raw_camera_frame=self._raw_frame
                )
            time.sleep(0.01)

    def show_frame(self):
        # input is raw_frame
        try:
            frame = self._raw_frame.copy()
            right_menu = self._raw_right_menu.copy()
        except AttributeError:
            pass
        # check the landmarks
        try:
            self._landmarks = self._pose_detection_results.pose_landmarks.landmark
            if self._copy_pose_runs:
                try:
                    own_landmarks = None
                    own_landmarks = self.read_snapshot(id_to_read=self._pose_id_read)
                except IndexError:
                    print("You copied the all pose. Congratulation")
                    self._copy_pose_runs = False
                if own_landmarks is not None and self.draw_the_body_landmarks(
                    saved_landmarks=own_landmarks,
                    camera_frame=frame,
                    fresh_landmarks=self._landmarks,
                ):
                    self._pose_id_read += 1
                    print(f"Congratulation you copied {self._pose_id_read} poses")

            if self._snapshot:
                self.record_snapshot(landmarks=self._landmarks)
                
            if self._macarena_runs or self._need_restart_macarena:
                self.prepare_macarena(camera_frame=frame, menu_frame=right_menu)

        except (KeyError, AttributeError):
            pass

        # Display frames in main program
        canvas = self.concat_camera_frame_with_menu(
            camera_frame=frame, right_bar=right_menu
        )
        cv2.imshow(self.PROGRAM_NAME, canvas)
        key = cv2.waitKey(1)
        if key == ord("q"):
            self.exit_program()
        if key == ord("m"):
            if not self._macarena_runs:
                self._macarena_runs = True
                self._time_start = time.time()
                self._current_score = 0
                self._need_restart_macarena = True
        if key == ord("s"):
            self._snapshot = True

        # TODO make it cleaner than this, global is 'clever' way
        global make_snapshot_by_mouse_click
        if make_snapshot_by_mouse_click:
            self._snapshot = True
            make_snapshot_by_mouse_click = False

        if key == ord("c"):
            if not self._copy_pose_runs:
                self._copy_pose_runs = True
                self._pose_id_read = 0
                self._need_restart_macarena = False
        if key == ord("r"):
            self._need_restart_macarena = False

    def prepare_macarena(self, camera_frame, menu_frame):
        menu_frame = self.draw_scores(
            right_bar=menu_frame,
            max_score=self._highest_score,
            current_score=self._current_score,
        )

        if self._macarena_runs:
            menu_frame = self.draw_instruction(right_bar=menu_frame)

            camera_frame = self.draw_action_hand(
                camera_frame=camera_frame,
                action_hand=self._action_hand,
                landmarks=self._landmarks,
            )
            camera_frame = self.draw_base_body_part(
                camera_frame=camera_frame,
                target_body_part=self._base_part,
                landmarks=self._landmarks,
            )
            elapsed = self.TIME_LIMIT_PER_ROUND - round(
                time.time() - self._time_start
            )
            if elapsed < 0:
                self._macarena_runs = False
            menu_frame = self.show_timer(right_bar=menu_frame, elapsed=elapsed)
            if self.done_what_simon_said(
                active_hand=self._action_hand,
                target_body_part=self._base_part,
                landmarks=self._landmarks,
                camera_frame=camera_frame,
            ):
                (
                    self._action_hand,
                    self._base_part,
                ) = self.regenerate_simon_instruction()
                self._current_score += 1
                if self._current_score > self._highest_score:
                    self._highest_score = self._current_score
        
    def record_snapshot(self, landmarks):
        self._snapshot = False
        with open(self.PATH_TO_SAVED_POSES, "r") as file:
            data = json.load(file)

        pose_time_snap = {"time": round(time.time())}
        pose_list = []
        for idx, coords in enumerate(landmarks):
            coords_dict = MessageToDict(coords)
            pose_list.append(coords_dict)
        pose_time_snap["landmarks"] = pose_list

        data.append(pose_time_snap)

        with open(self.PATH_TO_SAVED_POSES, "w") as file:
            json.dump(data, file, indent=4)

    def read_snapshot(self, id_to_read):
        with open(self.PATH_TO_SAVED_POSES, "r") as file:
            data = json.load(file)
            landmarks = data[id_to_read]["landmarks"]
            return landmarks

    def show_timer(self, right_bar, elapsed):
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        pos_x = 10
        pos_y = 300
        color = (0, 0, 255)  # red
        str_to_display = f"Remained {elapsed}s"
        right_bar = cv2.putText(
            right_bar, str_to_display, (pos_x, pos_y), font, 1, color, 1, cv2.LINE_AA
        )
        return right_bar

    def read_highest_score_from_file(self, path):
        try:
            with open(path, "r") as file:
                high_score = file.readline()
        except FileNotFoundError:
            return 0

        try:
            high_score = int(high_score)
            return high_score
        except ValueError:
            return 0

    def draw_instruction(self, right_bar):
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        pos_x = 10
        pos_y = 150
        color = (0, 0, 0)  # black

        right_bar = cv2.putText(
            right_bar,
            "To score a point",
            (pos_x, pos_y),
            font,
            1,
            color,
            1,
            cv2.LINE_AA,
        )

        pos_y = 170
        right_bar = cv2.putText(
            right_bar, "connect", (pos_x, pos_y), font, 1, color, 1, cv2.LINE_AA
        )

        right_bar = cv2.circle(
            right_bar,
            (pos_x + (self.ACTION_RADIUS), pos_y + (2 * self.ACTION_RADIUS)),
            self.ACTION_RADIUS,
            (0, 255, 0),
            -1,
        )

        right_bar = cv2.putText(
            right_bar,
            "to",
            (pos_x + (2 * self.ACTION_RADIUS + 10), pos_y + (2 * self.ACTION_RADIUS)),
            font,
            1,
            color,
            1,
            cv2.LINE_AA,
        )

        right_bar = cv2.circle(
            right_bar,
            (pos_x + (5 * self.ACTION_RADIUS), pos_y + (2 * self.ACTION_RADIUS)),
            self.ACTION_RADIUS,
            (255, 0, 0),
            -1,
        )

        return right_bar

    def draw_scores(self, right_bar, max_score, current_score):
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        pos_x = 10
        pos_y = 50
        color = (255, 0, 0)  # blue

        str_current_score = f"Current Score: {current_score}"
        right_bar = cv2.putText(
            right_bar, str_current_score, (pos_x, pos_y), font, 1, color, 2, cv2.LINE_AA
        )

        str_highest_score = f"Highest Score: {max_score}"
        right_bar = cv2.putText(
            right_bar,
            str_highest_score,
            (pos_x, pos_y + 50),
            font,
            1,
            color,
            1,
            cv2.LINE_AA,
        )

        return right_bar

    def exit_program(self):
        self.safe_the_high_score(self.PATH_TO_HIGH_SCORE)
        self._capture.release()
        cv2.destroyAllWindows()
        exit(0)

    def safe_the_high_score(self, path):
        try:
            with open(path, "r") as reader:
                high_score = reader.readline()
        except FileNotFoundError:
            return

        try:
            high_score = int(high_score)
            if self._highest_score > high_score:
                try:
                    with open(path, "w") as writer:
                        writer.write(str(self._highest_score))
                except FileNotFoundError:
                    return
        except ValueError:
            return

    def show_welcome(self):
        # Create right menu
        self._raw_right_menu = self.generate_simple_menu()
        
        # Create a green background image
        background_color = (0, 255, 0)  # Green
        background_image = 255 * np.ones((self.CAMERA_FRAME_HEIGHT, self.CAMERA_FRAME_WIDTH, 3), dtype=np.uint8)
        background_image[:] = background_color # mask it

        # Add title to the image
        text = "Hello in Macarena Game"
        text_position = (int(self.CAMERA_FRAME_WIDTH/2 - len(text)*6), int(self.CAMERA_FRAME_HEIGHT/2))
        text_color = (0, 0, 0)  # Black
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        cv2.putText(background_image, text, text_position, font, font_scale, text_color, thickness, cv2.LINE_AA)
        # Add warning to the image
        text = "Waiting for the camera input..."
        text_position = (int(self.CAMERA_FRAME_WIDTH/2 - len(text)*6), int(self.CAMERA_FRAME_HEIGHT/2 + 50))
        cv2.putText(background_image, text, text_position, font, font_scale, text_color, thickness, cv2.LINE_AA)
        
        # read to self._raw_frame because this picture need to be display until the camera frame is ready
        self._raw_frame = background_image
        if self._raw_frame is not None and self._raw_right_menu is not None:
            canvas = self.concat_camera_frame_with_menu(
                camera_frame=self._raw_frame, right_bar=self._raw_right_menu
            )
            cv2.imshow(self.PROGRAM_NAME, canvas)
        else:
            exit(1)

    def generate_simple_menu(self):
        # Create a gray background image
        background_color = (128, 128, 128)  # Gray
        background_image = 255 * np.ones((self.MENU_FRAME_HEIGHT, self.MENU_FRAME_WIDTH, 3), dtype=np.uint8)
        background_image[:] = background_color # mask it

        # Add title to the image
        text = "Menu"
        text_position_y = self.MENU_FRAME_HEIGHT // 2 + 100
        text_position_x = 5
  
        text_color = (0, 0, 0)  # Black
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        cv2.putText(background_image, text, (text_position_x, text_position_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
        
        # generate options
        font_scale = 0.45
        thickness = 1
        text_position_y += 20
        text = "Press 'm' to start the Macarena"
        cv2.putText(background_image, text, (text_position_x, text_position_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
        
        text_position_y += 15
        text = "Press 'c' to start the CopyGame"
        cv2.putText(background_image, text, (text_position_x, text_position_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
        
        text_position_y += 15
        text = "Press 's' to add pose snapshot"
        cv2.putText(background_image, text, (text_position_x, text_position_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
        
        text_position_y += 15
        text = "Press 'r' to restart"
        cv2.putText(background_image, text, (text_position_x, text_position_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
        
        text_position_y += 15
        text = "Press 'q' to quite the game"
        cv2.putText(background_image, text, (text_position_x, text_position_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
        
        
        # add author note
        font_scale = 0.35
        text_position_y += 25
        text = "Created by Marcin Naczk"
        cv2.putText(background_image, text, (text_position_x, text_position_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
        
        return background_image
        
    def concat_camera_frame_with_menu(self, camera_frame, right_bar):
        window_width = right_bar.shape[1] + camera_frame.shape[1]
        canvas = np.zeros(
            (max(right_bar.shape[0], camera_frame.shape[0]), window_width, 3),
            dtype=np.uint8,
        )
        # Place the camera frame on the left side of the canvas
        canvas[: camera_frame.shape[0], : camera_frame.shape[1]] = camera_frame
        # Place the menu bar on the right side of the canvas
        canvas[: right_bar.shape[0], camera_frame.shape[1] :] = right_bar
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

    def draw_the_body_landmarks(self, saved_landmarks, fresh_landmarks, camera_frame):
        correct_body_part = 0
        for pose_landmark in (
            self.REACHABLE_BY_BOTH_HANDS
            + self.REACHABLE_BY_LEFT_HAND
            + self.REACHABLE_BY_RIGHT_HAND
        ):
            id = pose_landmark.value
            name = pose_landmark.name
            single_saved_landmark = saved_landmarks[id]
            color = (255, 255, 255)  # white
            radius = round(self.ACTION_RADIUS / 4)
            x_saved = round(single_saved_landmark["x"] * camera_frame.shape[1])
            y_saved = round(single_saved_landmark["y"] * camera_frame.shape[0])

            single_fresh_landmark = fresh_landmarks[id]
            x_fresh = round(single_fresh_landmark.x * camera_frame.shape[1])
            y_fresh = round(single_fresh_landmark.y * camera_frame.shape[0])

            distance = math.dist([x_saved, y_saved], [x_fresh, y_fresh])
            if distance < (2 * self.ACTION_RADIUS):
                color = (0, 255, 0)  # green
                correct_body_part += 1

            cv2.circle(camera_frame, (x_saved, y_saved), radius, color, -1)
            font = cv2.FONT_HERSHEY_COMPLEX_SMALL
            cv2.putText(
                camera_frame,
                name,
                (x_saved + 5, y_saved + 5),
                font,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        return correct_body_part > 5

    def get_body_part_window_coordinate(self, body_index, landmarks, camera_frame):
        body_part = landmarks[body_index.value]
        x = round(body_part.x * camera_frame.shape[1])
        y = round(body_part.y * camera_frame.shape[0])
        return x, y

    def draw_action_hand(self, action_hand, camera_frame, landmarks):
        x, y = self.get_body_part_window_coordinate(
            body_index=action_hand, landmarks=landmarks, camera_frame=camera_frame
        )
        cv2.circle(camera_frame, (x, y), self.ACTION_RADIUS, (0, 255, 0), -1)
        return camera_frame

    def draw_base_body_part(self, target_body_part, camera_frame, landmarks):
        x, y = self.get_body_part_window_coordinate(
            body_index=target_body_part, landmarks=landmarks, camera_frame=camera_frame
        )
        cv2.circle(camera_frame, (x, y), self.ACTION_RADIUS, (255, 0, 0), -1)
        return camera_frame

    def done_what_simon_said(
        self, active_hand, target_body_part, landmarks, camera_frame
    ):
        active_x, active_y = self.get_body_part_window_coordinate(
            body_index=active_hand, landmarks=landmarks, camera_frame=camera_frame
        )
        target_x, target_y = self.get_body_part_window_coordinate(
            body_index=target_body_part, landmarks=landmarks, camera_frame=camera_frame
        )

        distance = math.dist([active_x, active_y], [target_x, target_y])

        return distance < self.ACTION_RADIUS

    def regenerate_simon_instruction(self):
        active_hand = random.choice(self.HANDS)

        if active_hand == self.MP_POSE.LEFT_INDEX:
            target_body_part = random.choice(
                self.REACHABLE_BY_BOTH_HANDS + self.REACHABLE_BY_LEFT_HAND
            )
        elif active_hand == self.MP_POSE.RIGHT_INDEX:
            target_body_part = random.choice(
                self.REACHABLE_BY_BOTH_HANDS + self.REACHABLE_BY_RIGHT_HAND
            )

        return active_hand, target_body_part


make_snapshot_by_mouse_click = False


def mouse_func(event, x, y, flags, param):
    global make_snapshot_by_mouse_click
    if event == cv2.EVENT_RBUTTONDOWN:
        make_snapshot_by_mouse_click = True
