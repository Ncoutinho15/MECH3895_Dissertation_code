# Import Library
import os
import numpy as np
import customtkinter as ct
from PIL import Image
from datetime import datetime
import time

#
from ExerciseTracker import ExerciseTracker
from ExercisePlanner import ExercisePlanner
from Chatbot import Chatbot

class RehabPalApp:
    def __init__(self):
        self.app = ct.CTk()
        self.app.geometry("800x480")
        #self.app.geometry("{0}x{1}+0+0".format(self.app.winfo_screenwidth(), self.app.winfo_screenheight()))
        self.app.title("RehabPal")
        ct.set_appearance_mode("light")

        #Background frame
        # Create a frame that fills the window and acts as a background
        self.background_frame = ct.CTkFrame(self.app, fg_color="#ffffff")
        self.background_frame.pack(fill="both", expand=True)

        # Header frame
        self.header = ct.CTkFrame(master=self.background_frame, fg_color="#99ddff", height=100)
        self.header.pack(fill="x")

        # User icon
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.user_icon = ct.CTkImage(Image.open(os.path.join(script_dir, "user.png")), size=(50, 50))
        self.label_image = ct.CTkButton(self.header, image=self.user_icon, corner_radius= None, text="", command=lambda: print("User icon clicked"), fg_color="#99ddff")
        self.label_image.pack(side=ct.LEFT)

        # Settings icon
        self.settings_icon = ct.CTkImage(Image.open(os.path.join(script_dir, "gear.png")), size=(50, 50))
        self.button_image = ct.CTkButton(self.header, image=self.settings_icon, corner_radius= None, text="", command=lambda: print("Settings clicked"), fg_color="#99ddff")
        self.button_image.pack(side=ct.RIGHT)

        # Date/time label
        self.date_time = datetime.now().strftime("%A  %H:%M \n %d/%m/%Y")
        self.label_date = ct.CTkLabel(self.header, text=self.date_time, font=("Roboto", 15), text_color="dark blue")
        self.label_date.pack(side=ct.RIGHT, padx=5)

        # Welcome label
        self.welcome_label = ct.CTkLabel(master=self.header, text="Welcome Nathan!", font=("Roboto", 30), text_color="dark blue")
        self.welcome_label.place(relx=0.5, rely=0.55, anchor=ct.CENTER)

        # Transparent frame to hold buttons
        self.button_frame = ct.CTkFrame(master=self.background_frame, fg_color="transparent")
        self.button_frame.pack(pady=120, padx=60)

        # Home button widgets
        self.button1 = ct.CTkButton(master=self.button_frame, text="Exercise\nPlanner", width=130, height=100, font=("Roboto", 40), fg_color="#0077bb", command=self.show_exercise_planner)
        self.button1.pack(side=ct.LEFT, padx=30)

        self.button2 = ct.CTkButton(master=self.button_frame, text="Exercise\nTracker", width=130, height=100, font=("Roboto", 40), fg_color="#0077bb", command=self.show_exercise_tracker)
        self.button2.pack(side=ct.LEFT, padx=30)

        self.button3 = ct.CTkButton(master=self.button_frame, text="Rehab\nChatbot", width=130, height=100, font=("Roboto", 40), fg_color="#0077bb", command=self.show_rehab_chatbot)
        self.button3.pack(side=ct.LEFT, padx=30)

        self.app.mainloop()
    
    # Functions to show each page
    def show_exercise_planner(self):
        print("Exercise Planner button clicked")
        self.background_frame.pack_forget()  # Hide the background frame

        self.exercise_planner = ExercisePlanner(self.app, home_callback=self.show_home_exercise_planner)
        self.exercise_planner.show()
    
    def show_exercise_tracker(self):
        print("Exercise Tracker button clicked")
        self.background_frame.pack_forget()  # Hide the background frame

        self.exercise_tracker = ExerciseTracker(self.app, home_callback=self.show_home_exercise_tracker)
        self.exercise_tracker.show()

    def show_rehab_chatbot(self):
        print("Rehab Chatbot button clicked")
        self.background_frame.pack_forget()  # Hide the background frame

        self.chatbot_page = Chatbot(self.app, home_callback=self.show_home_chatbot)
        self.chatbot_page.show()
    
    # Callback functions to return to home page
    def show_home_exercise_planner(self):
        print("Returning to home page")
        self.exercise_planner.hide()  # Hide the exercise planner page
        self.background_frame.pack(fill="both", expand=True)

    def show_home_exercise_tracker(self):
        print("Returning to home page")
        self.exercise_tracker.hide()  # Hide the exercise tracker page
        self.background_frame.pack(fill="both", expand=True)
        
    def show_home_chatbot(self):
        print("Returning to home page")
        self.chatbot_page.hide()  # Hide the chatbot page
        self.background_frame.pack(fill="both", expand=True)

if __name__ == "__main__":
    RehabPalApp()