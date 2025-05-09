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
    
    