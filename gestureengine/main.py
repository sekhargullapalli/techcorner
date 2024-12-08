import cv2
import mediapipe as mp

model_path = './gesture_recognizer.task'
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

overlay_gestures = []

def gesture_recognizer_callback(result, output_image, timestamp_ms):
    global overlay_gestures
    overlay_gestures = []
    
    if result.gestures:        
        gestures = []
        for hand_index, hand_gesture in enumerate(result.gestures):
            if hand_gesture:
                gesture = hand_gesture[0]  # Get the most confident gesture
                gestures.append((hand_index, gesture.category_name, gesture.score))
                overlay_gestures.append(f"Hand {hand_index + 1}: {gesture.category_name} ({gesture.score:.2f})")
                print(f"Hand {hand_index + 1}: {gesture.category_name} with confidence {gesture.score:.2f}")
            else:
                overlay_gestures.append(f"Hand {hand_index + 1}: No recognizable gesture")
                print(f"Hand {hand_index + 1}: None.")

        # Perform actions based on recognized gestures
        if len(gestures) == 1:
            single_hand_action(gestures[0][1])  
        elif len(gestures) == 2:
            dual_hand_action(gestures[0][1], gestures[1][1]) 
    else:
        print("No hands detected.")


def single_hand_action(gesture):
    print(f"Action for single hand: {gesture}")
    if gesture == "thumbs_up":
        print("Single-hand action: Thumbs Up!")

def dual_hand_action(gesture1, gesture2):
    print(f"Action for both hands: {gesture1} + {gesture2}")
    if gesture1 == "thumbs_up" and gesture2 == "victory":
        print("Dual-hand action: Thumbs Up + Victory!")

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    num_hands=2, min_hand_detection_confidence=0.7, min_tracking_confidence=0.7,
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=gesture_recognizer_callback
)

# Main script to process webcam feed
with GestureRecognizer.create_from_options(options) as recognizer:
    # Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        recognizer.recognize_async(mp_image, int(cv2.getTickCount() / cv2.getTickFrequency() * 1000))

        for i, gesture_text in enumerate(overlay_gestures):
            cv2.putText(
                frame,
                gesture_text,
                (10, 30 + i * 30),  # Position: 10 pixels from the left, line spacing of 30 pixels
                cv2.FONT_HERSHEY_SIMPLEX,  # Font type
                0.8,  # Font scale
                (0, 255, 0),  # Text color (green)
                2,  # Thickness
                cv2.LINE_AA  # Line type
            )

        cv2.imshow('Gesture Recognition', frame)
        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
