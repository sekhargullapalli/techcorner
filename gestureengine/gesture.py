import cv2
import mediapipe as mp
import time

# Define gestures
GESTURES = ["thumbs_up", "thumbs_down", "victory", "stop", "open_hand", "fist", "peace_sign"]

# Mediapipe Hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Placeholder for gesture recognition (replace with your ML logic or rules)
def recognize_gesture(landmarks):
    # Simulate recognition logic here. Replace with your own gesture recognition rules.
    # Example: Check thumb and finger positions to determine gestures.
    # This function should return a recognized gesture from GESTURES or None.
    return "open_hand"  # Example return gesture for testing

# Actions for single-hand gestures
def single_hand_action(gesture):
    print(f"Single: {gesture}")
    # Add your action logic here
    if gesture == "thumbs_up":
        print("Perform action for thumbs up.")
    elif gesture == "stop":
        print("Perform action for stop.")   

# Actions for dual-hand gestures
def dual_hand_action(gesture1, gesture2):
    print(f"Dual-: {gesture1} + {gesture2}")
    # Add your action logic here
    if gesture1 == "thumbs_up" and gesture2 == "victory":
        print("Perform action for thumbs up + victory.")

# Main loop to process video feed
def main():
    cap = cv2.VideoCapture(0)  # Open webcam
    delay = 0.1  # Configurable delay (in seconds)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame. Exiting...")
                break

            # Flip the frame to avoid mirror image
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame
            results = hands.process(rgb_frame)

            gestures = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Recognize gesture for each hand
                    gesture = recognize_gesture(hand_landmarks)
                    gestures.append(gesture)

                    # Draw hand landmarks on the frame
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Perform actions based on gestures
            if len(gestures) == 1:
                single_hand_action(gestures[0])
            elif len(gestures) == 2:
                dual_hand_action(gestures[0], gestures[1])

            # Display the video feed
            cv2.imshow("Gesture Recognition", frame)

            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(delay)

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

