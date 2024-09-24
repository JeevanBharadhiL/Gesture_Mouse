import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

tracking_enabled = True
last_x, last_y = 0, 0

# Function to check if index finger is closed
def index_finger_closed(hand_landmarks):
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
    return index_tip.y > index_dip.y

# Function to check if fist is closed (all fingers folded)
def fist_closed(hand_landmarks):
    return (hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y and
            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y and
            hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y and
            hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y and
            hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y)

# Function to check if index and middle fingers are open while others are closed
def index_and_middle_open(hand_landmarks):
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_dip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_dip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_dip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP]
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    
    return (index_tip.y < index_dip.y and 
            middle_tip.y < middle_dip.y and
            ring_tip.y > ring_dip.y and 
            pinky_tip.y > pinky_dip.y and 
            thumb_tip.y > thumb_ip.y)

# Function to draw landmarks with different colors
def draw_landmarks(frame, hand_landmarks):
    for landmark in mp_hands.HandLandmark:
        landmark_point = hand_landmarks.landmark[landmark]
        landmark_x = int(landmark_point.x * frame.shape[1])
        landmark_y = int(landmark_point.y * frame.shape[0])

        if landmark == mp_hands.HandLandmark.INDEX_FINGER_TIP:
            color = (0, 255, 0)  # Green for index finger tip
        elif landmark == mp_hands.HandLandmark.MIDDLE_FINGER_TIP:
            color = (255, 0, 0)  # Blue for middle finger tip
        else:
            color = (255, 0, 255)  # Magenta for other landmarks

        cv2.circle(frame, (landmark_x, landmark_y), 5, color, -1)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and get hand landmarks
    results = hands.process(frame_rgb)

    # Track index finger and move mouse cursor
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            draw_landmarks(frame, hand_landmarks)

            # Get the index finger tip position
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_x = int(index_finger_tip.x * frame.shape[1])
            index_y = int(index_finger_tip.y * frame.shape[0])

            if index_finger_closed(hand_landmarks):
                tracking_enabled = False
            elif fist_closed(hand_landmarks):
                pyautogui.click(last_x, last_y)
            elif index_and_middle_open(hand_landmarks):
                pyautogui.doubleClick(last_x, last_y)
            else:
                tracking_enabled = True

            if tracking_enabled:
                # Move the mouse cursor to the index finger tip position
                screen_width, screen_height = pyautogui.size()
                mouse_x = int(index_finger_tip.x * screen_width)
                mouse_y = int(index_finger_tip.y * screen_height)
                pyautogui.moveTo(mouse_x, mouse_y)
                last_x, last_y = mouse_x, mouse_y

    # Display the frame with landmarks
    cv2.imshow('Index Finger Tracking', frame)

    # Break the loop when 'Esc' key is pressed
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
