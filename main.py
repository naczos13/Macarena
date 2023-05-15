import cv2
import mediapipe as mp
import numpy as np


def main():
    # Set up the mediapipe instances
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    
    capture = cv2.VideoCapture(0)

    ## Tune mediapipe detection and tracking
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while capture.isOpened():
            ret, frame = capture.read()
            
            # Recolor image because mediapipe need RGB, 
            # and cv2 has default BGR
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Memory optimization
            image.flags.writeable = False
            
            # Make detection
            results = pose.process(image)
            
            # Memory optimization
            image.flags.writeable = True
            # Recolor image back
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                # Clear face landmarks
                CUTOFF_THRESHOLD = 10  # head and face according to https://camo.githubusercontent.com/7fbec98ddbc1dc4186852d1c29487efd7b1eb820c8b6ef34e113fcde40746be2/68747470733a2f2f6d65646961706970652e6465762f696d616765732f6d6f62696c652f706f73655f747261636b696e675f66756c6c5f626f64795f6c616e646d61726b732e706e67
                for landmark_id, landmark in enumerate(landmarks):
                    if landmark_id <= CUTOFF_THRESHOLD:
                        landmark.visibility = 0
                
                
                # Get coordinates
                left_shoulder       = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                left_elbow          = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                left_wrist          = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                
                right_shoulder      = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                right_elbow         = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                right_wrist         = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                
                # Calculate angle
                left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                
                # Vizualize left angle
                cv2.putText(image, str(left_angle),
                            tuple(np.multiply([left_elbow.x, left_elbow.y], [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
                
                # Vizualize lright angle
                cv2.putText(image, str(right_angle),
                            tuple(np.multiply([right_elbow.x, right_elbow.y], [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
                
                
            except KeyError:
                pass
            
            generate_text_with_instructions(image, "The Simon Says!")
            genereate_text_with_score(image, "SCORE: 25")
            generate_text_with_helper(image, "Helper: is not ok")
            
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)           
            
            cv2.imshow('Simple Tracker', image)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            
        capture.release()
        cv2.destroyAllWindows()
    
    
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
    #TODO change the color dependig if the user is in proper position
    font_color = (0, 0, 255)  # RED color

    # Left Top Corner
    text_x = 20
    text_y = 50

    # Put the text on the image
    cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, font_color, 2)
    


if __name__ == "__main__":
    main()


