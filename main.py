import cv2
import mediapipe as mp
import helper


def main():
    # create main window
    PROGRAM_NAME = "Simon Says"
    cv2.namedWindow(PROGRAM_NAME)

    # get input from camera
    capture = cv2.VideoCapture(0)

    ## Tune mediapipe detection and tracking
    pose_detector = mp.solutions.pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )

    while capture.isOpened():
        ret, frame = capture.read()

        # Recolor image because mediapipe need RGB,
        # and cv2 has default BGR
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Memory optimization
        image.flags.writeable = False

        # Make detection
        results = pose_detector.process(image)

        # Memory optimization
        image.flags.writeable = True
        # Recolor image back
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            helper.disable_head_landmarks(landmarks)

            helper.display_arm_angles(image, landmarks)

            helper.draw_action_hand(image, landmarks, "left_wrist")

        except (KeyError, AttributeError):
            print("some error")
            pass

        helper.generate_text_with_instructions(image, "The Simon Says!")
        helper.genereate_text_with_score(image, "SCORE: 25")
        helper.generate_text_with_helper(image, "Helper: is not ok")

        # Render detections
        mp.solutions.drawing_utils.draw_landmarks(
            image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS
        )

        cv2.imshow(PROGRAM_NAME, image)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
