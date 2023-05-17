import cv2
import mediapipe as mp
import helper


def main():
    # create main window
    PROGRAM_NAME = "Simon Says"
    cv2.namedWindow(PROGRAM_NAME)
    cv2.setMouseCallback(PROGRAM_NAME, helper.mouse_click_event)

    # get input from camera
    capture = cv2.VideoCapture(0)
    action_part = helper.regenerate_simon_instruction()

    while capture.isOpened():
        ret, frame = capture.read()
        results = helper.find_human_pose_in_image(frame)     
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            helper.update_layout(frame, landmarks)
                      
            # Render detections
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS
            )

        except (KeyError, AttributeError):
            print("some error")
            pass

        cv2.imshow(PROGRAM_NAME, frame)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
