def update_processing_visualization(self, processed_images):
        """Update the processing steps visualization"""
        # Clear all axes
        for ax in self.axes:
            ax.clear()
            ax.axis('off')
        
        # Set titles
        self.axes[0].set_title('RGB')
        self.axes[1].set_title('Grayscale')
        self.axes[2].set_title('Blurred')
        self.axes[3].set_title('Thresholded')
        self.axes[4].set_title('Contours')
        self.axes[5].set_title('Computer\'s Choice')
        
        # Display images
        self.axes[0].imshow(processed_images['rgb'])
        self.axes[1].imshow(processed_images['gray'], cmap='gray')
        self.axes[2].imshow(processed_images['blurred'], cmap='gray')
        self.axes[3].imshow(processed_images['threshold'], cmap='gray')
        self.axes[4].imshow(cv2.cvtColor(processed_images['contours'], cv2.COLOR_BGR2RGB))
        self.axes[5].imshow(cv2.cvtColor(processed_images['computer_choice'], cv2.COLOR_BGR2RGB))
        
        # Update the canvas
        self.fig.tight_layout()
        self.canvas.draw()
    
    def start_countdown(self):
        """Start the countdown for the game"""
        # Reset game state
        self.user_gesture = None
        self.computer_gesture = None
        self.result = None
        self.countdown_active = True
        self.countdown_value = 3
        self.result_label.config(text="Get ready to show your gesture!")
        self.countdown_label.config(text=str(self.countdown_value))
        
        # Start the countdown timer
        self.root.after(1000, self.update_countdown)
    
    def update_countdown(self):
        """Update the countdown timer"""
        self.countdown_value -= 1
        
        if self.countdown_value >= 0:
            self.countdown_label.config(text=str(self.countdown_value))
            if self.countdown_value == 0:
                self.countdown_label.config(text="SHOOT!")
            self.root.after(1000, self.update_countdown)
        else:
            # After "SHOOT!" is displayed for a second, clear it
            self.countdown_label.config(text="")
