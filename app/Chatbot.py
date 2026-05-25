import customtkinter as ct
from PIL import Image
from datetime import datetime
import os
from LLM_Engine import RehabLLM
import threading


class Chatbot:
    def __init__(self, master, home_callback=None):
        self.master = master
        self.home_callback = home_callback
        self.frame = ct.CTkFrame(master, fg_color="#f0f4f7")
        self.create_widgets()
        self.llm = RehabLLM()

    def create_widgets(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Use grid so the bottom entry area always remains visible.
        self.frame.grid_rowconfigure(0, weight=0)  # Header
        self.frame.grid_rowconfigure(1, weight=1)  # Chat area
        self.frame.grid_rowconfigure(2, weight=0)  # Input bar
        self.frame.grid_columnconfigure(0, weight=1)

        # ================= HEADER =================
        header_frame = ct.CTkFrame(
            self.frame,
            fg_color="#99ddff",
            height=76,
            corner_radius=0,
        )
        header_frame.grid(row=0, column=0, sticky="ew")
        header_frame.grid_propagate(False)

        self.home_icon = ct.CTkImage(
            Image.open(os.path.join(script_dir, "home.png")),
            size=(44, 44),
        )

        self.home_button = ct.CTkButton(
            header_frame,
            image=self.home_icon,
            text="",
            width=64,
            height=60,
            command=self.on_home_clicked,
            fg_color="#99ddff",
            hover_color="#99ddff",
        )
        self.home_button.pack(side="left", padx=6, pady=8)

        self.title_label = ct.CTkLabel(
            header_frame,
            text="Rehab Chatbot",
            font=("Roboto", 30),
            text_color="dark blue",
        )
        self.title_label.place(relx=0.5, rely=0.5, anchor=ct.CENTER)

        self.settings_icon = ct.CTkImage(
            Image.open(os.path.join(script_dir, "gear.png")),
            size=(44, 44),
        )

        self.button_image = ct.CTkButton(
            header_frame,
            image=self.settings_icon,
            text="",
            width=64,
            height=60,
            command=lambda: print("Settings clicked"),
            fg_color="#99ddff",
            hover_color="#99ddff",
        )
        self.button_image.pack(side=ct.RIGHT, padx=6, pady=8)

        self.date_time = datetime.now().strftime("%A %H:%M\n%d/%m/%Y")
        self.label_date = ct.CTkLabel(
            header_frame,
            text=self.date_time,
            font=("Roboto", 14),
            text_color="dark blue",
        )
        self.label_date.pack(side=ct.RIGHT, padx=4)

        # ================= CHAT AREA =================
        self.chat_area = ct.CTkScrollableFrame(
            self.frame,
            fg_color="#eef7ff",
            corner_radius=12,
            label_text="",
        )
        self.chat_area.grid(
            row=1,
            column=0,
            sticky="nsew",
            padx=10,
            pady=(8, 6),
        )

        self.chat_area.grid_columnconfigure(0, weight=1)

        # Optional welcome message so the page does not look empty.
        self.add_bubble(
            "Hello! Ask me about your shoulder or arm rehabilitation.",
            "#d2f2ff",
            "w",
        )

        # ================= BOTTOM INPUT BAR =================
        bottom_frame = ct.CTkFrame(
            self.frame,
            fg_color="#dfe7ef",
            height=62,
            corner_radius=0,
        )
        bottom_frame.grid(row=2, column=0, sticky="ew")
        bottom_frame.grid_propagate(False)

        bottom_frame.grid_columnconfigure(0, weight=1)
        bottom_frame.grid_columnconfigure(1, weight=0)

        self.message_entry = ct.CTkEntry(
            bottom_frame,
            placeholder_text="Type your message...",
            height=42,
            font=("Roboto", 15),
        )
        self.message_entry.grid(
            row=0,
            column=0,
            sticky="ew",
            padx=(10, 8),
            pady=10,
        )
        #added for demonstration
        self.message_entry.insert(0, "My upper shoulder has a stinging pain")

        self.message_entry.bind("<Return>", lambda event: self.on_send())

        self.send_button = ct.CTkButton(
            bottom_frame,
            text="Send",
            width=82,
            height=42,
            font=("Roboto", 15, "bold"),
            fg_color="#0077bb",
            hover_color="#005f99",
            command=self.on_send,
        )
        self.send_button.grid(
            row=0,
            column=1,
            padx=(0, 10),
            pady=10,
        )

    def add_bubble(self, text, color, anchor):
        bubble = ct.CTkLabel(
            self.chat_area,
            text=text,
            fg_color=color,
            text_color="black",
            corner_radius=14,
            justify="left",
            wraplength=520,
            padx=12,
            pady=10,
            font=("Roboto", 14),
        )
        bubble.pack(anchor=anchor, pady=5, padx=8)

        # Auto-scroll to bottom.
        self.chat_area.update_idletasks()
        try:
            self.chat_area._parent_canvas.yview_moveto(1.0)
        except Exception:
            pass

    def display_response(self, response):
        self.is_loading = False

        if hasattr(self, "loading_label"):
            self.loading_label.destroy()

        self.add_bubble(response, "#d2f2ff", "w")

    def animate_loading(self):
        dots = ["Loading.", "Loading..", "Loading..."]

        def loop(i=0):
            if self.is_loading and hasattr(self, "loading_label"):
                self.loading_label.configure(text=dots[i % 3])
                self.master.after(500, loop, i + 1)

        loop()

    def on_send(self):
        message = self.message_entry.get().strip()
        if not message:
            return

        self.add_bubble(message, "#99ddff", "e")
        self.message_entry.delete(0, ct.END)

        if hasattr(self, "loading_label"):
            self.loading_label.destroy()

        self.is_loading = True

        self.loading_label = ct.CTkLabel(
            self.chat_area,
            text="Loading.",
            font=("Roboto", 14),
            text_color="black",
        )
        self.loading_label.pack(anchor="w", padx=8, pady=5)

        self.animate_loading()

        threading.Thread(
            target=self.run_llm,
            args=(message,),
            daemon=True,
        ).start()

    def run_llm(self, message):
        try:
            response = self.llm.chat(message)
        except Exception as e:
            response = f"Error: {str(e)}"

        self.master.after(0, lambda: self.display_response(response))

    def on_home_clicked(self):
        if callable(self.home_callback):
            self.home_callback()

    def show(self):
        self.frame.place(relx=0, rely=0, relwidth=1, relheight=1)

    def hide(self):
        self.frame.place_forget()