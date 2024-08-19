import cv2
from ultralytics import YOLO
import numpy as np
import time
from gtts import gTTS
from playsound import playsound
import tempfile

# Load the YOLO models
ball_model = YOLO("best.pt")
pose_model = YOLO("yolov8m-pose.pt")

# Open the webcam
cap = cv2.VideoCapture(0)

# Initialize counters and positions
dribble_count = 0
step_count = 0
prev_x_center = None
prev_y_center = None
prev_left_ankle_y = None
prev_right_ankle_y = None
prev_delta_y = None
ball_not_detected_frames = 0
max_ball_not_detected_frames = 20  # Adjust based on your requirement
dribble_threshold = 18  # Adjust based on observations
step_threshold = 5
min_wait_frames = 7
wait_frames = 0
travel_detected = False
travel_timestamp = None
total_dribble_count = 0
total_step_count = 0

# Set delay for detecting the next travel after the current one
travel_delay = 5  # 5 seconds delay

# Define the body part indices
body_index = {"left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16}

# Define the frame dimensions and fps
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

out = None

def play_travel_alert():
    tts = gTTS(text="Travel terdeteksi!", lang='id')
    with tempfile.NamedTemporaryFile(delete=True) as fp:
        tts.save(f"{fp.name}.mp3")
        playsound(f"{fp.name}.mp3")

while cap.isOpened():
    success, frame = cap.read()

    if success:

        # Ball detection
        ball_results_list = ball_model(frame, verbose=False, conf=0.65)

        ball_detected = False

        for results in ball_results_list:
            for bbox in results.boxes.xyxy:
                x1, y1, x2, y2 = bbox[:4]

                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2

                if prev_y_center is not None:
                    delta_y = y_center - prev_y_center

                    if (
                        prev_delta_y is not None
                        and prev_delta_y > dribble_threshold
                        and delta_y < -dribble_threshold
                    ):
                        dribble_count += 1
                        total_dribble_count += 1

                    prev_delta_y = delta_y

                prev_x_center = x_center
                prev_y_center = y_center

                ball_detected = True
                ball_not_detected_frames = 0

            annotated_frame = results.plot()

        # Increment the ball not detected counter if ball is not detected
        if not ball_detected:
            ball_not_detected_frames += 1

        # Reset step count if ball is not detected for a prolonged period
        if ball_not_detected_frames >= max_ball_not_detected_frames:
            step_count = 0

        # Pose detection
        pose_results = pose_model(frame, verbose=False, conf=0.5)

        # Round the results to the nearest decimal
        keypoints_data = pose_results[0].keypoints.data
        keypoints_array = np.array(keypoints_data)
        rounded_results = np.round(keypoints_array, 1) 

        try:
            left_knee = rounded_results[0][body_index["left_knee"]]
            right_knee = rounded_results[0][body_index["right_knee"]]
            left_ankle = rounded_results[0][body_index["left_ankle"]]
            right_ankle = rounded_results[0][body_index["right_ankle"]]

            if (
                (left_knee[2] > 0.5)
                and (right_knee[2] > 0.5)
                and (left_ankle[2] > 0.5)
                and (right_ankle[2] > 0.5)
            ):
                if (
                    prev_left_ankle_y is not None
                    and prev_right_ankle_y is not None
                    and wait_frames == 0
                ):
                    left_diff = abs(left_ankle[1] - prev_left_ankle_y)
                    right_diff = abs(right_ankle[1] - prev_right_ankle_y)

                    if max(left_diff, right_diff) > step_threshold:
                        step_count += 1
                        total_step_count += 1
                        print(f"Step taken: {step_count}")
                        wait_frames = min_wait_frames  # Update wait_frames

                prev_left_ankle_y = left_ankle[1]
                prev_right_ankle_y = right_ankle[1]

                if wait_frames > 0:
                    wait_frames -= 1

        except:
            print("No human detected.")

        pose_annotated_frame = pose_results[0].plot()

        # Combining frames
        combined_frame = cv2.addWeighted(
            annotated_frame, 0.6, pose_annotated_frame, 0.4, 0
        )

        # Drawing counts on the frame
        cv2.putText(
            combined_frame,
            f"Total dribble count: {total_dribble_count}",
            (100, 500),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        # Travel detection
        if ball_detected and step_count > 2 and dribble_count == 0:
            if not travel_detected or (time.time() - travel_timestamp > travel_delay):
                print("Travel terdeteksi!")
                step_count = 0  # reset step count
                travel_detected = True
                travel_timestamp = time.time()
                play_travel_alert()  # Play travel alert sound

        if travel_detected and time.time() - travel_timestamp > 10:
            travel_detected = False
            total_dribble_count = 0
            total_step_count = 0

        # Change the tint of the frame and write text if travel was detected
        if travel_detected:
            # Change the tint of the frame to blue
            blue_tint = np.full_like(combined_frame, (0, 0, 255), dtype=np.uint8)
            combined_frame = cv2.addWeighted(combined_frame, 0.7, blue_tint, 0.3, 0)

            # Write 'Travel Detected!' at the top right of the screen
            cv2.putText(
                combined_frame,
                "Travel terdeteksi!",
                (
                    frame_width - 600,
                    150,
                ),  # You might need to adjust these values
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 255, 255),
                4,
                cv2.LINE_AA,
            )

        # Reset counts when a dribble is detected
        if dribble_count > 0:
            step_count = 0
            dribble_count = 0

        cv2.imshow("Travel_Detection", combined_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Release the VideoWriter if it's still open
if out is not None:
    out.release()

cap.release()
cv2.destroyAllWindows()
