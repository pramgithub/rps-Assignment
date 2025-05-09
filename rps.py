import cv2
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from PIL import Image, ImageTk
import mediapipe as mp
from collections import deque

class RockPaperScissorsGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Rock Paper Scissors Game")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")
        
        # Initialize MediaPipe hands module
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Game state variables
        self.countdown_active = False
        self.countdown_value = 3
        self.user_gesture = None
        self.computer_gesture = None
        self.result = None
        self.score = {"user": 0, "computer": 0, "ties": 0}
        self.game_history = deque(maxlen=5)  # Store recent game results
        
        # Create a frame for the camera feed and processing steps
        self.camera_frame = tk.Frame(root, bg="#e0e0e0", padx=10, pady=10)
        self.camera_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create a frame for the game controls and results
        self.control_frame = tk.Frame(root, bg="#e0e0e0", width=400, padx=20, pady=20)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            exit()
        
        # Set up camera feed display - making it larger since it now contains the hand detection
        self.camera_label = tk.Label(self.camera_frame)
        self.camera_label.grid(row=0, column=0, padx=5, pady=5)
        
        # Set up image processing displays (now with more steps)
        self.fig, self.axes = plt.subplots(2, 3, figsize=(10, 6))
        self.fig.suptitle('Image Processing Steps', fontsize=16)
        self.axes = self.axes.flatten()
        
        for ax in self.axes:
            ax.axis('off')
        
        self.axes[0].set_title('RGB')
        self.axes[1].set_title('Grayscale')
        self.axes[2].set_title('Blurred')
        self.axes[3].set_title('Thresholded')
        self.axes[4].set_title('Contours')
        self.axes[5].set_title('Computer\'s Choice')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.camera_frame)
        self.canvas.get_tk_widget().grid(row=1, column=0, padx=5, pady=5)
        
        # Game controls
        self.title_label = tk.Label(self.control_frame, text="ROCK PAPER SCISSORS", font=("Arial", 18, "bold"), bg="#e0e0e0")
        self.title_label.pack(pady=10)
        
        self.countdown_label = tk.Label(self.control_frame, text="", font=("Arial", 36, "bold"), bg="#e0e0e0", fg="#ff5722")
        self.countdown_label.pack(pady=20)
        
        self.result_label = tk.Label(self.control_frame, text="Ready to play?", font=("Arial", 16), bg="#e0e0e0")
        self.result_label.pack(pady=10)
        
        self.score_label = tk.Label(self.control_frame, text="Score: You 0 - 0 Computer (Ties: 0)", font=("Arial", 12), bg="#e0e0e0")
        self.score_label.pack(pady=10)
        
        self.start_button = tk.Button(self.control_frame, text="Start Game", font=("Arial", 14), command=self.start_countdown, bg="#4caf50", fg="white", padx=20, pady=10)
        self.start_button.pack(pady=20)
        
        self.quit_button = tk.Button(self.control_frame, text="Quit", font=("Arial", 14), command=self.quit_game, bg="#f44336", fg="white", padx=20, pady=10)
        self.quit_button.pack(pady=10)
        
        # History section
        self.history_label = tk.Label(self.control_frame, text="Game History", font=("Arial", 14, "bold"), bg="#e0e0e0")
        self.history_label.pack(pady=(20, 10))
        
        self.history_text = tk.Text(self.control_frame, height=5, width=35, font=("Arial", 10))
        self.history_text.pack(pady=5)
        self.history_text.config(state=tk.DISABLED)
        
        # Start the video loop
        self.update_frame()
    ////
    
    def update_frame(self):
        """Update the camera feed and processing displays"""
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Failed to grab frame")
            return
        
        # Flip frame horizontally for a selfie-view
        frame = cv2.flip(frame, 1)
        original_frame = frame.copy()
        
        # Process the frame for hand gesture recognition
        processed_images, enhanced_frame = self.process_frame(frame)
        
        # Display the enhanced frame with hand detection on the main camera feed
        camera_image = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
        camera_image = Image.fromarray(camera_image)
        camera_image = ImageTk.PhotoImage(image=camera_image)
        self.camera_label.imgtk = camera_image
        self.camera_label.config(image=camera_image)
        
        # Update the processing steps visualization
        self.update_processing_visualization(processed_images)
        
        # If countdown is active, update it
        if self.countdown_active:
            self.countdown_label.config(text=str(self.countdown_value))
        
        # Schedule next frame update
        self.root.after(10, self.update_frame)
    
    def process_frame(self, frame):
        """Process the frame to detect hand gestures"""
        # Create a larger frame to show the hand detection
        enhanced_frame = frame.copy()
        
        # Convert to RGB for MediaPipe and visualization
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get grayscale image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # Apply thresholding
        _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Apply adaptive thresholding for better results
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_display = frame.copy()
        cv2.drawContours(contours_display, contours, -1, (0, 255, 0), 2)
        
        # MediaPipe hand detection
        results = self.hands.process(rgb_frame)
        detected_gesture = "None"
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on the enhanced frame
                self.mp_drawing.draw_landmarks(
                    enhanced_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Identify gesture
                detected_gesture = self.identify_gesture(hand_landmarks)
                
                # If the countdown is at 0 and we haven't stored the user's gesture yet
                if self.countdown_active and self.countdown_value == 0 and self.user_gesture is None:
                    self.user_gesture = detected_gesture
                    self.computer_gesture = self.get_computer_gesture()
                    self.determine_winner()
                    self.countdown_active = False
                    self.update_history()
        
        # Create computer's choice image
        computer_choice_img = self.create_computer_choice_image()
        
        # Add gesture information to the enhanced frame
        cv2.putText(
            enhanced_frame, 
            f"Detected: {detected_gesture}", 
            (10, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2, 
            cv2.LINE_AA
        )
        
        # Add countdown to the enhanced frame if active
        if self.countdown_active:
            cv2.putText(
                enhanced_frame,
                f"Countdown: {self.countdown_value if self.countdown_value > 0 else 'SHOOT!'}",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA
            )
        
        # Add game result to the enhanced frame if available
        if self.user_gesture is not None and self.result is not None:
            cv2.putText(
                enhanced_frame,
                f"You: {self.user_gesture} vs PC: {self.computer_gesture}",
                (10, enhanced_frame.shape[0] - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 165, 0),
                2,
                cv2.LINE_AA
            )
            
            cv2.putText(
                enhanced_frame,
                f"Result: {self.result}",
                (10, enhanced_frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 165, 0),
                2,
                cv2.LINE_AA
            )
        
        return {
            'rgb': rgb_frame,
            'gray': gray,
            'blurred': blurred,
            'threshold': threshold,
            'contours': contours_display,
            'computer_choice': computer_choice_img
        }, enhanced_frame
    