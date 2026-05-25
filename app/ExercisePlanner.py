import customtkinter as ct
from PIL import Image
from datetime import datetime
import os
import threading
import json

from ExercisePlannerEngine import ExercisePlannerEngine

try:
    from LLM_Engine import RehabLLM
except ImportError:
    RehabLLM = None


class ExercisePlanner:
    def __init__(self, master, home_callback=None):
        self.master = master
        self.home_callback = home_callback

        self.frame = ct.CTkFrame(master, fg_color="#f0f4f7")
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

        self.llm_engine = RehabLLM() if RehabLLM else None
        self.backend = ExercisePlannerEngine(llm_engine=self.llm_engine)

        self.current_programme_data = None
        self.exercise_buttons = []

        # Default profile used because the GUI input widgets have been removed.
        self.default_patient_data = {
            "age": "",
            "condition": "Shoulder Impingement",
            "stage": "early",
            "irritability": "high",
            "limitations": "",
        }

        self.create_widgets()

    def create_widgets(self):
        # ================= MAIN GRID =================
        # Fixed rows so everything fits inside 800x480.
        self.frame.grid_rowconfigure(0, weight=0)  # Header
        self.frame.grid_rowconfigure(1, weight=0)  # Generate button
        self.frame.grid_rowconfigure(2, weight=1)  # Main content
        self.frame.grid_rowconfigure(3, weight=0)  # Report button
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
            Image.open(os.path.join(self.script_dir, "home.png")),
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
            text="Exercise Planner",
            font=("Roboto", 34),
            text_color="dark blue",
        )
        self.title_label.place(relx=0.5, rely=0.5, anchor=ct.CENTER)

        self.settings_icon = ct.CTkImage(
            Image.open(os.path.join(self.script_dir, "gear.png")),
            size=(44, 44),
        )

        self.settings_button = ct.CTkButton(
            header_frame,
            image=self.settings_icon,
            text="",
            width=64,
            height=60,
            command=lambda: print("Settings clicked"),
            fg_color="#99ddff",
            hover_color="#99ddff",
        )
        self.settings_button.pack(side=ct.RIGHT, padx=6, pady=8)

        self.label_date = ct.CTkLabel(
            header_frame,
            text=datetime.now().strftime("%A %H:%M\n%d/%m/%Y"),
            font=("Roboto", 14),
            text_color="dark blue",
        )
        self.label_date.pack(side=ct.RIGHT, padx=4)

        # ================= GENERATE BUTTON =================
        self.generate_button = ct.CTkButton(
            self.frame,
            text="Generate Exercise Programme",
            font=("Roboto", 24),
            fg_color="#087fb3",
            hover_color="#066894",
            height=44,
            corner_radius=0,
            command=self.generate_programme,
        )
        self.generate_button.grid(
            row=1,
            column=0,
            sticky="ew",
            padx=8,
            pady=(8, 5),
        )

        # ================= CONTENT AREA =================
        content_frame = ct.CTkFrame(
            self.frame,
            fg_color="#f0f4f7",
            corner_radius=0,
        )
        content_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 4))

        # Previous GUI/reference design: table left, info panel right.
        # Left side is slightly wider so Exercise Name, Reps and Sets fit.
        content_frame.grid_columnconfigure(0, weight=56)
        content_frame.grid_columnconfigure(1, weight=44)
        content_frame.grid_rowconfigure(0, weight=1)

        self.left_frame = ct.CTkFrame(
            content_frame,
            fg_color="#f0f4f7",
            corner_radius=0,
        )
        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 3))

        self.right_frame = ct.CTkFrame(
            content_frame,
            fg_color="#16a0cf",
            corner_radius=0,
        )
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=(3, 0))

        # ================= LEFT: PROGRAMME TABLE =================
        self.programme_title = ct.CTkLabel(
            self.left_frame,
            text="Exercise Programme",
            font=("Roboto", 22, "bold"),
            fg_color="#07045c",
            text_color="white",
            height=42,
            corner_radius=0,
        )
        self.programme_title.pack(fill="x")

        self.table_frame = ct.CTkFrame(
            self.left_frame,
            fg_color="#cfe3f3",
            corner_radius=0,
        )
        self.table_frame.pack(fill="both", expand=True)

        self.table_frame.grid_columnconfigure(0, weight=1)
        self.table_frame.grid_columnconfigure(1, weight=0)
        self.table_frame.grid_columnconfigure(2, weight=0)

        # ================= RIGHT: EXERCISE INFO =================
        self.info_title = ct.CTkLabel(
            self.right_frame,
            text="Exercise Information",
            font=("Roboto", 22, "bold"),
            fg_color="#07045c",
            text_color="white",
            height=42,
            corner_radius=0,
        )
        self.info_title.pack(fill="x")

        self.info_textbox = ct.CTkTextbox(
            self.right_frame,
            fg_color="#16a0cf",
            text_color="black",
            font=("Roboto", 15),
            wrap="word",
            border_width=0,
            corner_radius=0,
        )
        self.info_textbox.pack(fill="both", expand=True, padx=6, pady=6)

        self.info_textbox.insert(
            "1.0",
            "Generate a programme, then select an exercise to view instructions.",
        )
        self.info_textbox.configure(state="disabled")

        # ================= REPORT BUTTON =================
        self.report_button = ct.CTkButton(
            self.frame,
            text="Save Physiotherapist PDF Report",
            font=("Roboto", 14),
            command=self.save_report,
            fg_color="#07045c",
            hover_color="#05033d",
            height=30,
            corner_radius=0,
        )
        self.report_button.grid(
            row=3,
            column=0,
            sticky="ew",
            padx=8,
            pady=(0, 5),
        )

        self.draw_empty_table()

    # ================= TABLE HELPERS =================

    def _configure_button_text_wrap(self, button, wraplength=250):
        try:
            button._text_label.configure(
                wraplength=wraplength,
                justify="left",
            )
        except Exception:
            pass

    def _draw_headers(self):
        headers = ["Exercise Name", "Reps", "Sets"]
        widths = [250, 58, 54]

        for col, header in enumerate(headers):
            label = ct.CTkLabel(
                self.table_frame,
                text=header,
                font=("Roboto", 16, "bold"),
                text_color="white",
                fg_color="#16a0cf",
                width=widths[col],
                height=36,
                anchor="w",
                padx=5,
                corner_radius=0,
            )
            label.grid(row=0, column=col, sticky="nsew", padx=1, pady=1)

    def draw_empty_table(self):
        for widget in self.table_frame.winfo_children():
            widget.destroy()

        self._draw_headers()

        for row in range(1, 6):
            bg = "#cfe3f3" if row % 2 else "#e9f4fb"

            name = ct.CTkLabel(
                self.table_frame,
                text=f"Exercise {row}",
                font=("Roboto", 15),
                fg_color=bg,
                text_color="black",
                height=34,
                anchor="w",
                padx=5,
                wraplength=245,
                justify="left",
                corner_radius=0,
            )
            name.grid(row=row, column=0, sticky="nsew", padx=1, pady=1)

            reps = ct.CTkLabel(
                self.table_frame,
                text="",
                font=("Roboto", 15),
                fg_color=bg,
                text_color="black",
                height=34,
                width=58,
                corner_radius=0,
            )
            reps.grid(row=row, column=1, sticky="nsew", padx=1, pady=1)

            sets = ct.CTkLabel(
                self.table_frame,
                text="",
                font=("Roboto", 15),
                fg_color=bg,
                text_color="black",
                height=34,
                width=54,
                corner_radius=0,
            )
            sets.grid(row=row, column=2, sticky="nsew", padx=1, pady=1)

    # ================= BACKEND LOGIC =================

    def load_patient_profile(self, filename="user.json"):
        path = os.path.join(self.script_dir, filename)

        if not os.path.exists(path):
            return dict(self.default_patient_data)

        try:
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)

            return {
                "age": str(data.get("age", "")).strip(),
                "condition": str(data.get("condition", self.default_patient_data["condition"])).strip(),
                "stage": str(data.get("stage", self.default_patient_data["stage"])).strip().lower(),
                "irritability": str(data.get("irritability", self.default_patient_data["irritability"])).strip().lower(),
                "limitations": str(data.get("limitations", "")).strip()
            }

        except Exception as e:
            self.set_info_text(f"Could not load user.json. Using default profile.\n\nError:\n{e}")
            return dict(self.default_patient_data)

    def generate_programme(self):
        patient_data = self.load_patient_profile()

        self.generate_button.configure(
            text="Generating...",
            state="disabled"
        )
        self.set_info_text(
            "Generating exercise programme...\n\n"
            "The local LLM may take a while on the Raspberry Pi."
        )

        threading.Thread(
            target=self._run_generate_programme,
            args=(patient_data,),
            daemon=True
        ).start()


    def _run_generate_programme(self, patient_data):
        try:
            programme_data = self.backend.generate_programme(patient_data)

            self.master.after(
                0,
                lambda: self._display_generated_programme(programme_data)
            )

        except Exception as e:
            self.master.after(
                0,
                lambda: self._display_generation_error(e)
            )


    def _display_generated_programme(self, programme_data):
        self.current_programme_data = programme_data
        self.populate_programme_table(programme_data["programme"])

        metadata = programme_data.get("generation_metadata", {})
        warnings = metadata.get("warnings", [])
        mode = metadata.get("selection_mode", "unknown")

        message = (
            f"Programme generated from user profile.\n"
            f"Generation mode: {mode}\n\n"
            "Select an exercise to view instructions."
        )

        if warnings:
            gui_warnings = [
            w for w in warnings
            if "LLM planner fallback triggered" not in w
        ]

        if gui_warnings:
            message += "\n\nWarnings:\n" + "\n".join([f"- {w}" for w in gui_warnings])

        self.set_info_text(message)

        self.generate_button.configure(
            text="Generate Exercise Programme",
            state="normal"
        )


    def _display_generation_error(self, error):
        self.set_info_text(f"Error generating programme:\n{error}")

        self.generate_button.configure(
            text="Generate Exercise Programme",
            state="normal"
        )

    def populate_programme_table(self, programme):
        for widget in self.table_frame.winfo_children():
            widget.destroy()

        self._draw_headers()
        self.exercise_buttons.clear()

        for row, exercise in enumerate(programme, start=1):
            bg = "#cfe3f3" if row % 2 else "#e9f4fb"

            button = ct.CTkButton(
                self.table_frame,
                text=exercise["name"],
                font=("Roboto", 14, "bold"),
                fg_color=bg,
                hover_color="#b8d8ed",
                text_color="black",
                height=34,
                anchor="w",
                corner_radius=0,
                command=lambda e=exercise: self.show_exercise_info(e),
            )
            button.grid(row=row, column=0, sticky="nsew", padx=1, pady=1)

            self._configure_button_text_wrap(button, wraplength=245)
            self.exercise_buttons.append(button)

            reps_label = ct.CTkLabel(
                self.table_frame,
                text=str(exercise["reps"]),
                font=("Roboto", 15),
                fg_color=bg,
                text_color="black",
                height=34,
                width=58,
                corner_radius=0,
            )
            reps_label.grid(row=row, column=1, sticky="nsew", padx=1, pady=1)

            sets_label = ct.CTkLabel(
                self.table_frame,
                text=str(exercise["sets"]),
                font=("Roboto", 15),
                fg_color=bg,
                text_color="black",
                height=34,
                width=54,
                corner_radius=0,
            )
            sets_label.grid(row=row, column=2, sticky="nsew", padx=1, pady=1)

    def show_exercise_info(self, exercise):
        hold_text = ""
        if exercise.get("hold_seconds"):
            hold_text = f"\nHold: {exercise['hold_seconds']} seconds"

        xai = exercise.get("xai_explanation", {})

        why_items = xai.get("selected_because", [])
        safety_items = xai.get("safety_checks", [])

        why_text = "\n".join([f"- {item}" for item in why_items])
        safety_text = "\n".join([f"- {item}" for item in safety_items])

        if not why_text:
            why_text = "- Selected based on the current rehabilitation profile."

        if not safety_text:
            safety_text = "- Stop if pain increases or symptoms worsen."

        text = (
            f"{exercise['name']}\n\n"
            f"Sets: {exercise['sets']}\n"
            f"Reps: {exercise['reps']}\n"
            f"Frequency: {exercise['frequency']}"
            f"{hold_text}\n\n"
            f"How to perform:\n{exercise['instructions']}\n\n"
            f"Avoid:\n{exercise['avoid']}\n\n"
            f"Why selected:\n{why_text}\n\n"
            f"Safety checks:\n{safety_text}"
        )

        self.set_info_text(text)

    def set_info_text(self, text):
        self.info_textbox.configure(state="normal")
        self.info_textbox.delete("1.0", "end")
        self.info_textbox.insert("1.0", text)
        self.info_textbox.configure(state="disabled")

    def save_report(self):
        if not self.current_programme_data:
            self.set_info_text("Generate an exercise programme before saving a report.")
            return

        try:
            path = self.backend.save_pdf_report(self.current_programme_data)

            if path:
                self.set_info_text(f"PDF report saved:\n{path}")
            else:
                self.set_info_text(
                    "PDF report could not be saved. Install reportlab using: pip install reportlab"
                )
        except Exception as e:
            self.set_info_text(f"Error saving PDF report:\n{e}")

    # ================= NAVIGATION =================

    def on_home_clicked(self):
        if callable(self.home_callback):
            self.home_callback()

    def show(self):
        self.frame.place(relx=0, rely=0, relwidth=1, relheight=1)

    def hide(self):
        self.frame.place_forget()