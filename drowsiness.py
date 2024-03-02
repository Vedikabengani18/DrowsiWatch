import cv2  # Import the OpenCV library
import winsound  # Import the winsound library for playing sound
from imutils import face_utils  # Import face_utils from the imutils library
import dlib  # Import the dlib library
from scipy.spatial import distance as dist  # Import the distance module from scipy.spatial
import imutils  # Import the imutils library

# Define the DrowsinessDetection class
class DrowsinessDetection:
    def __init__(self):
        # Initialize variables and setup drowsiness detection components
        self.frequency = 2500  # Frequency of the beep sound
        self.duration = 1000  # Duration of the beep sound
        self.count = 0  # Counter for tracking consecutive frames with drowsiness
        self.ear_thresh = 0.3  # Threshold value for eye aspect ratio to detect drowsiness
        self.ear_frames = 48  # Number of consecutive frames with low eye aspect ratio to consider drowsiness
        self.shape_predictor = "shape_predictor_68_face_landmarks.dat"  # Path to the facial landmarks predictor model

        self.cam = cv2.VideoCapture(0)  # Initialize the webcam capture
        self.detector = dlib.get_frontal_face_detector()  # Initialize the face detector from dlib
        self.predictor = dlib.shape_predictor(self.shape_predictor)  # Initialize the facial landmarks predictor

        # Define the indices for left and right eyes in the facial landmarks
        (self.l_start, self.l_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.r_start, self.r_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # Method to calculate the eye aspect ratio
    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    # Method to detect drowsiness
    def detect_drowsiness(self):

        while True:
            _, frame = self.cam.read()  # Read a frame from the webcam
            frame = imutils.resize(frame, width=450)  # Resize the frame to a smaller width
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale

            rects = self.detector(gray, 0)  # Detect faces in the grayscale frame

            for rect in rects:
                shape = self.predictor(gray, rect)  # Predict facial landmarks
                shape = face_utils.shape_to_np(shape)  # Convert facial landmarks to NumPy array

                left_eye = shape[self.l_start:self.l_end]  # Extract left eye region from facial landmarks
                right_eye = shape[self.r_start:self.r_end]  # Extract right eye region from facial landmarks
                left_ear = self.eye_aspect_ratio(left_eye)  # Calculate left eye aspect ratio
                right_ear = self.eye_aspect_ratio(right_eye)  # Calculate right eye aspect ratio

                ear = (left_ear + right_ear) / 2.0  # Average of both eye aspect ratios

                left_eye_hull = cv2.convexHull(left_eye)  # Compute the convex hull for left eye
                right_eye_hull = cv2.convexHull(right_eye)  # Compute the convex hull for right eye
                cv2.drawContours(frame, [left_eye_hull], -1, (0, 0, 255), 1)  # Draw the contours for left eye
                cv2.drawContours(frame, [right_eye_hull], -1, (0, 0, 255), 1)  # Draw the contours for right eye

                if ear < self.ear_thresh:  # Check if the eye aspect ratio is below the threshold
                    self.count += 1  # Increment the drowsiness counter

                    if self.count >= self.ear_frames:  # Check if the drowsiness has persisted for enough frames
                        cv2.putText(frame, "DROWSINESS DETECTED", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  # Add text to indicate drowsiness detection
                        winsound.Beep(self.frequency, self.duration)  # Play a beep sound to alert of drowsiness
                else:
                    self.count = 0  # Reset the drowsiness counter

            cv2.imshow("Frame", frame)  # Display the frame with detected landmarks and contours
            key = cv2.waitKey(1) & 0xFF  # Wait for a key press and mask it to get the ASCII value

            if key == ord("q"):  # Check if the 'q' key is pressed
                break  # Exit the loop if 'q' key is pressed

        self.cam.release()  # Release the webcam
        cv2.destroyAllWindows()  # Close all OpenCV windows

# Instantiate the DrowsinessDetection class and start drowsiness detection
from drowsiness import DrowsinessDetection
if __name__=="__main__":
    detect_drowsine =DrowsinessDetection()  # Create an instance of the DrowsinessDetection class
    detect_drowsine.detect_drowsiness()  # Start the drowsiness detection process
