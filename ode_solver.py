import customtkinter as ctk
from sympy import symbols, Function, Eq, dsolve, sympify, Derivative, lambdify, Float
from sympy.printing import pretty
from PIL import Image, ImageTk
import os
import sys
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from io import BytesIO
import tkinter as tk

# --- BEGIN: resource_path function for PyInstaller compatibility ---
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)
# --- END: resource_path ---

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

LANG = {
    "en": {
        "main_title": "Differential Equation Solver",
        "desc": "Enter the order and coefficients of your ODE on the left.\n"
                "Each coefficient corresponds to the derivative order as labeled.\n"
                "You can solve both homogeneous and non-homogeneous equations.",
        "order_label": "Order of Equation (n):",
        "set_order": "Set Order",
        "restart": "Restart",
        "fx_label": "f(x) (right-hand side, enter 0 for homogeneous):",
        "solve": "Solve",
        "stop": "Stop",
        "status_solving": "Solving...",
        "status_stopped": "Stopped.",
        "status_order_err": "Order must be a positive integer.",
        "status_coeff_err": "Error: All coefficients must be numbers.",
        "status_order_int_err": "Error: Order must be an integer.",
        "status_fx_err": "The input expression could not be interpreted correctly.",
        "status_plot_range_err": "Please enter valid x-range.",
        "status_plot_failed": "Failed to plot solution.",
        "status_plot_const": "Please enter value for {const}.",
        "status_dirfield_range_err": "Please enter a valid x-range.",
        "status_dirfield_failed": "Failed to evaluate slope field.",
        "status_dirfield_parse": "Failed to parse ODE for field.",
        "answer": "Result",
        "plot": "Plotter",
        "plot_btn_text": "Plot",
        "plot_solution": "Plot Solution",
        "dirfield": "Directional Field",
        "log": "Log",
        "back": "Back",
        "help": "Help / Manual",
        "language": "Language:",
        "answer_is": "The answer is:\n\n",
        "solving_log": "Solving Log",
        "no_log": "No log available. Solve an equation first.",
        "manual": """Manual

• How to enter an ODE:
  - Enter the order (n) as a positive integer.
  - Enter coefficients from highest derivative to y (use decimal numbers or integers).
  - Enter f(x) as the right-hand side (0 for homogeneous equations). Example: sin(x), exp(-2*x), x**2+3.

• Solve:
  - Click 'Solve'. The equation and solution will appear on the answer tab.

• Plotter:
  - If the answer contains constants (C1, C2, ...), fill their values.
  - Set the x range and click 'Plot'.

• Directional Field:
  - Only available for first-order ODEs.
  - Enter x range and click 'Show Directional Field'.

• Log:
  - See the step-by-step solving process.

• Restart:
  - Click 'Restart' to reset the program.

Notes:
- Coefficients must be numbers.
- Only non-symbolic right-hand sides are allowed (f(x) only in terms of x).
""",
        "manual_title": "Help / Manual"
    },
    "fa": {
        "main_title": "حلگر معادله دیفرانسیل",
        "desc": "در سمت چپ، مرتبه و ضرایب معادله دیفرانسیل خود را وارد کنید.\n"
                "هر ضریب مربوط به مرتبه مشتق برچسب‌گذاری شده است.\n"
                "می‌توانید معادلات همگن یا ناهمگن را حل کنید.",
        "order_label": "مرتبه معادله (n):",
        "set_order": "تایید مرتبه",
        "restart": "شروع مجدد",
        "fx_label": "f(x) (سمت راست معادله، برای همگن صفر وارد کنید):",
        "solve": "حل کن",
        "stop": "توقف",
        "status_solving": "در حال حل...",
        "status_stopped": "متوقف شد.",
        "status_order_err": "مرتبه باید یک عدد صحیح مثبت باشد.",
        "status_coeff_err": "خطا: همه ضرایب باید عددی باشند.",
        "status_order_int_err": "خطا: مرتبه باید عدد صحیح باشد.",
        "status_fx_err": "عبارت وارد شده برای f(x) قابل تفسیر نیست.",
        "status_plot_range_err": "بازه x را به درستی وارد کنید.",
        "status_plot_failed": "رسم راه‌حل ناموفق بود.",
        "status_plot_const": "مقدار {const} را وارد کنید.",
        "status_dirfield_range_err": "بازه x را به درستی وارد کنید.",
        "status_dirfield_failed": "محاسبه میدان شیب ناموفق بود.",
        "status_dirfield_parse": "تجزیه معادله برای میدان ناموفق بود.",
        "answer": "پاسخ",
        "plot": "رسم",
        "plot_btn_text": "رسم",
        "plot_solution": "رسم راه‌حل",
        "dirfield": "میدان جهتی",
        "log": "گزارش",
        "back": "بازگشت",
        "help": "راهنما / دستورالعمل",
        "language": "زبان:",
        "answer_is": "پاسخ:\n\n",
        "solving_log": "گزارش حل",
        "no_log": "گزارشی وجود ندارد. ابتدا معادله‌ای حل کنید.",
        "manual": """دستورالعمل

• نحوه وارد کردن معادله دیفرانسیل:
  - مرتبه (n) را به صورت عدد صحیح مثبت وارد کنید.
  - ضرایب را از بالاترین مرتبه تا y وارد کنید (عدد صحیح یا اعشاری).
  - f(x) را برای سمت راست وارد کنید (برای همگن، صفر وارد کنید). مثال: sin(x)، exp(-2*x)، x**2+3.

• حل:
  - روی 'حل کن' کلیک کنید. معادله و پاسخ در تب پاسخ نمایش داده می‌شود.

• رسم:
  - اگر جواب دارای ثابت (C1، C2 و ...) بود، مقدار آن را وارد کنید.
  - بازه x را مشخص کرده و روی 'رسم' کلیک کنید.

• میدان جهتی:
  - فقط برای معادلات مرتبه اول فعال است.
  - بازه x را وارد و روی 'نمایش میدان جهتی' کلیک کنید.

• گزارش:
  - مراحل حل را مشاهده کنید.

• شروع مجدد:
  - برای بازنشانی برنامه روی 'شروع مجدد' کلیک کنید.

نکات:
- ضرایب باید عددی باشند.
- فقط عبارات عددی برای f(x) قابل قبول است (فقط بر حسب x).
""",
        "manual_title": "راهنما / دستورالعمل"
    }
}

def prettify_floats(expr):
    def _pretty_float(x):
        val = float(x)
        r = round(val, 2)
        if r.is_integer():
            return int(r)
        else:
            return Float(r, 2)
    return expr.xreplace({f: _pretty_float(f) for f in expr.atoms(Float)})

class ODEGui(ctk.CTk):
    def __init__(self):
        super().__init__()
        try:
            logo_path = resource_path("logo.png")
            if os.path.exists(logo_path):
                logo_img = Image.open(logo_path)
                logo_tk = ImageTk.PhotoImage(logo_img)
                self.wm_iconphoto(True, logo_tk)
                self.icon_tk = logo_tk
        except Exception as e:
            print("Could not set window icon:", e)

        self.lang = "en"
        self.title("KNTU ODE Solver")
        self.geometry("850x600")
        self.minsize(850, 600)
        self.ode_order = 1
        self.log_text = ""
        self.answer_img = None
        self.answer_img_buf = None
        self.answer_pil_img = None
        self.solving_thread = None
        self._stop_solve = threading.Event()
        self.buttons_navbar = []

        self.input_page = ctk.CTkFrame(self)
        self.input_page.pack(fill="both", expand=True)
        self.result_page = ctk.CTkFrame(self)
        self.result_page.pack_forget()
        self.help_page = ctk.CTkFrame(self)
        self.help_page.pack_forget()

        self.build_input_page()
        self.build_result_page()
        self.build_help_tab()
        self.update_language()
        
        self.all_images = []

    def build_input_page(self):
        self.top_frame = ctk.CTkFrame(self.input_page)
        self.top_frame.pack(side="top", fill="x", pady=(10, 2), padx=10)
        self.left_frame = ctk.CTkFrame(self.top_frame)
        self.left_frame.pack(side="left", anchor="n", fill="both", expand=True)
        self.right_frame = ctk.CTkFrame(self.top_frame, width=260)
        self.right_frame.pack(side="right", anchor="n", fill="y", padx=10, pady=10)
        try:
            image = Image.open(resource_path("logo.png")).resize((120, 120))
            self.logo_image = ctk.CTkImage(light_image=image, dark_image=image, size=(120, 120))
            self.logo_label = ctk.CTkLabel(self.right_frame, image=self.logo_image, text="")
            self.logo_label.pack(pady=(10, 7))
        except Exception:
            self.logo_label = ctk.CTkLabel(self.right_frame, text="(No Logo)", font=ctk.CTkFont(size=18, weight="bold"))
            self.logo_label.pack(pady=(10, 7))

        self.detail_title = ctk.CTkLabel(self.right_frame, text="")
        self.detail_title.pack(pady=(2, 0), fill="x")

        self.detail_paragraph = ctk.CTkLabel(self.right_frame, text="", wraplength=240)
        self.detail_paragraph.pack(pady=(8, 0), padx=8, fill="x")

        self.button_help = ctk.CTkButton(self.right_frame, text="", fg_color="gray", hover_color="#444444", command=self.show_help_page)
        self.button_help.pack(pady=(14, 2), padx=8, fill="x")

        # Language changer
        lang_frame = ctk.CTkFrame(self.right_frame, fg_color="transparent")
        lang_frame.pack(pady=(8, 4), padx=8, fill="x")
        self.lang_label = ctk.CTkLabel(lang_frame, text="", width=66)
        self.lang_label.pack(side="left", padx=(0,4))
        self.lang_var = ctk.StringVar(value="English")
        self.lang_combo = ctk.CTkComboBox(lang_frame, values=["English", "فارسی"], command=self.change_language, variable=self.lang_var, width=96, state="readonly")
        self.lang_combo.pack(side="left", padx=(0,2))

        self.order_container = ctk.CTkFrame(self.left_frame, fg_color="transparent")
        self.order_container.pack(fill="x", pady=(5, 8))
        self.label_n = ctk.CTkLabel(self.order_container, text="")
        self.label_n.pack(side="left", pady=(2, 2), padx=(0, 8))
        self.entry_n = ctk.CTkEntry(self.order_container, width=80)
        self.entry_n.pack(side="left", padx=(0, 8))
        self.button_set_order = ctk.CTkButton(self.order_container, text="", command=self.create_coefficient_entries)
        self.button_set_order.pack(side="left", pady=6, padx=(0, 8))
        self.button_restart = ctk.CTkButton(self.order_container, text="", command=self.restart_program, fg_color="gray", hover_color="#555555")
        self.button_restart.pack(side="left", pady=6, padx=(0, 8))

        self.coeff_outer_frame = ctk.CTkFrame(self.left_frame)
        self.coeff_outer_frame.pack(fill="x", pady=(0, 14), padx=2)
        self.coeff_scrollable = ctk.CTkScrollableFrame(self.coeff_outer_frame, height=280)
        self.coeff_scrollable.pack(fill="x", padx=6, pady=8)
        self.coeff_entries = []

        self.fx_container = ctk.CTkFrame(self.left_frame, fg_color="transparent")
        self.fx_container.pack(fill="x", pady=(5, 5))
        self.label_fx = ctk.CTkLabel(self.fx_container, text="")
        self.label_fx.pack(anchor="w", pady=(2, 2), padx=(0, 8))
        self.entry_fx = ctk.CTkEntry(self.fx_container)
        self.entry_fx.pack(side="left", padx=(0, 8))
        self.button_solve = ctk.CTkButton(self.fx_container, text="", command=self.start_solve)
        self.button_solve.pack(side="left", pady=8, padx=(0, 8))
        self.button_stop = ctk.CTkButton(self.fx_container, text="", command=self.stop_solve, fg_color="gray", hover_color="#888888")
        self.button_stop.pack(side="left", pady=8, padx=(0, 8))
        self.button_stop.configure(state="disabled")

        self.status_label = ctk.CTkLabel(self.left_frame, text="", font=ctk.CTkFont(size=14, weight="bold"), text_color="red")
        self.status_label.pack(anchor="center", pady=(2, 2))

    def build_result_page(self):
        self.result_padding = ctk.CTkFrame(self.result_page)
        self.result_padding.pack(fill="both", expand=True, padx=24, pady=20)

        self.result_navbar = ctk.CTkFrame(self.result_padding)
        self.result_navbar.pack(fill="x", pady=(10, 0))
        self.answer_btn = ctk.CTkButton(self.result_navbar, text="", command=lambda: self.show_result_section("answer"))
        self.plot_btn = ctk.CTkButton(self.result_navbar, text="", command=lambda: self.show_result_section("plotter"))
        self.dirfield_btn = ctk.CTkButton(self.result_navbar, text="", command=lambda: self.show_result_section("dirfield"))
        self.log_btn = ctk.CTkButton(self.result_navbar, text="", command=lambda: self.show_result_section("log"), fg_color="gray", hover_color="#444444")
        self.answer_btn.pack(side="left", padx=8)
        self.plot_btn.pack(side="left", padx=8)
        self.dirfield_btn.pack(side="left", padx=8)
        self.log_btn.pack(side="left", padx=8)
        
        self.back_button = ctk.CTkButton(self.result_navbar, text="", command=self.rerun_program, fg_color="gray", hover_color="#444444")
        self.back_button.pack(side="right", padx=(0, 8))
        self.buttons_navbar = [self.answer_btn, self.plot_btn, self.dirfield_btn, self.log_btn]

        # Frame for answer
        self.answer_frame = ctk.CTkFrame(self.result_padding)
        self.answer_label = ctk.CTkLabel(self.answer_frame, text="", font=ctk.CTkFont(size=20, weight="bold"))
        self.answer_label.pack(pady=(18,6))
        self.answer_img_label = ctk.CTkLabel(self.answer_frame, text="")
        self.answer_img_label.pack(pady=(6, 2))
        self.answer_text = ctk.CTkTextbox(self.answer_frame, height=130, width=680, wrap="word", font=ctk.CTkFont(size=15))
        self.answer_text.pack(pady=6, padx=20, fill="both", expand=False)

        # Frame for plotter
        self.plotter_frame = ctk.CTkFrame(self.result_padding)
        self.plotter_label = ctk.CTkLabel(self.plotter_frame, text="", font=ctk.CTkFont(size=19, weight="bold"))
        self.plotter_label.pack(pady=(18, 3))
        self.consts_frame = ctk.CTkFrame(self.plotter_frame, fg_color="transparent")
        self.consts_frame.pack(pady=(2,2))
        self.const_entries = []
        self.range_label = ctk.CTkLabel(self.plotter_frame, text="")
        self.range_label.pack(side="left", padx=(18,2))
        self.range_x0_entry = ctk.CTkEntry(self.plotter_frame, width=70)
        self.range_x0_entry.pack(side="left", padx=2)
        self.range_label2 = ctk.CTkLabel(self.plotter_frame, text="")
        self.range_label2.pack(side="left", padx=(2,2))
        self.range_x1_entry = ctk.CTkEntry(self.plotter_frame, width=70)
        self.range_x1_entry.pack(side="left", padx=2)
        self.plot_button = ctk.CTkButton(self.plotter_frame, text=LANG[self.lang]["plot_btn_text"], command=self.plot_solution)
        self.plot_button.pack(pady=(12,8))
        self.plot_canvas_frame = ctk.CTkFrame(self.plotter_frame)
        self.plot_canvas_frame.pack(pady=10, padx=24, fill="both", expand=True)
        self.plot_canvas = None

        # Frame for directional field
        self.dirfield_frame = ctk.CTkFrame(self.result_padding)
        self.dirfield_label = ctk.CTkLabel(self.dirfield_frame, text="", font=ctk.CTkFont(size=19, weight="bold"))
        self.dirfield_label.pack(pady=(18, 8))
        self.dirfield_xrange_frame = ctk.CTkFrame(self.dirfield_frame, fg_color="transparent")
        self.dirfield_xrange_frame.pack(pady=(2,2))
        self.dirfield_x0_label = ctk.CTkLabel(self.dirfield_xrange_frame, text="")
        self.dirfield_x0_label.pack(side="left", padx=(2,2))
        self.dirfield_x0_entry = ctk.CTkEntry(self.dirfield_xrange_frame, width=70)
        self.dirfield_x0_entry.pack(side="left", padx=2)
        self.dirfield_x1_label = ctk.CTkLabel(self.dirfield_xrange_frame, text="")
        self.dirfield_x1_label.pack(side="left", padx=(2,2))
        self.dirfield_x1_entry = ctk.CTkEntry(self.dirfield_xrange_frame, width=70)
        self.dirfield_x1_entry.pack(side="left", padx=2)
        self.dirfield_button = ctk.CTkButton(self.dirfield_frame, text="", command=self.plot_directional_field)
        self.dirfield_button.pack(pady=6)
        self.dirfield_canvas_frame = ctk.CTkFrame(self.dirfield_frame)
        self.dirfield_canvas_frame.pack(pady=10, padx=24, fill="both", expand=True)
        self.dirfield_canvas = None

        # Frame for log
        self.log_frame = ctk.CTkFrame(self.result_padding)
        self.log_label = ctk.CTkLabel(self.log_frame, text="", font=ctk.CTkFont(size=19, weight="bold"))
        self.log_label.pack(pady=(18,6))
        self.log_textbox = ctk.CTkTextbox(self.log_frame, font=("Consolas", 13), wrap="word", width=680, height=250)
        self.log_textbox.pack(padx=20, pady=(5,16), fill="both", expand=True)
        self.log_textbox.configure(state="disabled")

    def build_help_tab(self):
        self.help_frame = ctk.CTkFrame(self.help_page)
        self.help_frame.pack(fill="both", expand=True, padx=32, pady=32)
        self.help_title = ctk.CTkLabel(self.help_frame, text="", font=ctk.CTkFont(size=20, weight="bold"))
        self.help_title.pack(pady=(8, 12))

        # Use classic Tkinter Text for help/manual to support justify
        self.help_tab_textbox = tk.Text(self.help_frame, font=("Vazirmatn", 13), wrap="word", width=48, height=19, borderwidth=0)
        self.help_tab_textbox.pack(padx=12, pady=(8, 20), fill="both", expand=True)
        self.help_tab_textbox.configure(state="disabled", bg="#212121", fg="#f6f6f6", relief="flat", highlightthickness=0)

        self.button_help_back = ctk.CTkButton(self.help_frame, text="", command=self.hide_help_page, fg_color="gray", hover_color="#444444")
        self.button_help_back.pack(pady=(0, 8), fill="x")

    def rerun_program(self):
        import subprocess, sys, os
        python = sys.executable
        script = os.path.abspath(sys.argv[0])
        subprocess.Popen([python, script] + sys.argv[1:])
        sys.exit(0)


    def update_language(self, *_):
        lang = self.lang
        align = "e" if lang == "fa" else "w"
        justify = "right" if lang == "fa" else "left"
        font_title = ctk.CTkFont(size=18, weight="bold", family="Vazirmatn" if lang == "fa" else "Arial")

        self.detail_title.configure(text=LANG[lang]["main_title"], anchor=align, justify=justify, font=font_title)
        self.detail_paragraph.configure(text=LANG[lang]["desc"], anchor=align, justify=justify)
        self.button_help.configure(text=LANG[lang]["help"], anchor=align)
        self.lang_label.configure(text=LANG[lang]["language"], anchor=align)
        self.plot_button.configure(text=LANG[lang]["plot_btn_text"])

        self.label_n.configure(text=LANG[lang]["order_label"], anchor=align)
        self.button_set_order.configure(text=LANG[lang]["set_order"])
        self.button_restart.configure(text=LANG[lang]["restart"])
        self.label_fx.configure(text=LANG[lang]["fx_label"], anchor=align)
        self.button_solve.configure(text=LANG[lang]["solve"])
        self.button_stop.configure(text=LANG[lang]["stop"])

        self.answer_btn.configure(text=LANG[lang]["answer"])
        self.plot_btn.configure(text=LANG[lang]["plot"])
        self.dirfield_btn.configure(text=LANG[lang]["dirfield"])
        self.log_btn.configure(text=LANG[lang]["log"])
        self.back_button.configure(text=LANG[lang]["back"])

        self.answer_label.configure(text=LANG[lang]["answer"])
        self.plotter_label.configure(text=LANG[lang]["plot_solution"])
        self.range_label.configure(text=LANG[lang]["fx_label"] if lang == "fa" else "x range: from")
        self.range_label2.configure(text="تا" if lang == "fa" else "to")
        self.dirfield_label.configure(text=LANG[lang]["dirfield"])
        self.dirfield_x0_label.configure(text="بازه x: از" if lang == "fa" else "x range: from")
        self.dirfield_x1_label.configure(text="تا" if lang == "fa" else "to")
        self.dirfield_button.configure(text="نمایش میدان جهتی" if lang == "fa" else "Show Directional Field")

        self.log_label.configure(text=LANG[lang]["solving_log"])
        self.button_help_back.configure(text=LANG[lang]["back"])
        self.help_title.configure(text=LANG[lang]["manual_title"])

        # Manual/help text
        self.help_tab_textbox.configure(state="normal")
        self.help_tab_textbox.delete("1.0", tk.END)
        self.help_tab_textbox.insert("1.0", LANG[lang]["manual"])
        self.help_tab_textbox.configure(state="disabled")
        self.help_tab_textbox.tag_configure("tag-right", justify=justify)
        self.help_tab_textbox.tag_add("tag-right", "1.0", tk.END)

        # Direction for the rest of widgets (frames, alignments)
        for widget in [self.detail_title, self.detail_paragraph, self.label_n, self.label_fx]:
            widget.configure(anchor=align, justify=justify)

        # For coeff labels if they exist
        for widget in getattr(self, "coeff_entries", []):
            if hasattr(widget, "master"):
                for child in widget.master.winfo_children():
                    if isinstance(child, ctk.CTkLabel):
                        child.configure(anchor=align, justify=justify)

        # Flip layout direction if needed
        if lang == "fa":
            self.left_frame.pack_forget()
            self.left_frame.pack(side="right", anchor="n", fill="both", expand=True)
            self.right_frame.pack_forget()
            self.right_frame.pack(side="left", anchor="n", fill="y", padx=10, pady=10)
        else:
            self.right_frame.pack_forget()
            self.right_frame.pack(side="right", anchor="n", fill="y", padx=10, pady=10)
            self.left_frame.pack_forget()
            self.left_frame.pack(side="left", anchor="n", fill="both", expand=True)

    def change_language(self, event=None):
        lang_map = {"English": "en", "فارسی": "fa"}
        self.lang = lang_map.get(self.lang_combo.get(), "en")
        self.update_language()

    def show_help_page(self):
        self.input_page.pack_forget()
        self.result_page.pack_forget()
        self.help_page.pack(fill="both", expand=True)
        # Disable navbar buttons
        for btn in self.buttons_navbar:
            btn.configure(state="disabled")

    def hide_help_page(self):
        self.help_page.pack_forget()
        self.input_page.pack(fill="both", expand=True)
        # Re-enable navbar buttons
        for btn in self.buttons_navbar:
            btn.configure(state="normal")

    def show_result_page(self, solution_text, order, sol=None, rhs=None, ode_expr=None):
        self.input_page.pack_forget()
        self.help_page.pack_forget()
        self.result_page.pack(fill="both", expand=True)
        self.show_result_section("answer")
        self.ode_order = order
        self.solution_expr = sol
        self.ode_rhs = rhs
        self.ode_expr = ode_expr
        self.answer_text.delete("1.0", ctk.END)
        self.answer_text.insert(ctk.END, LANG[self.lang]["answer_is"] + solution_text)

        for widget in self.consts_frame.winfo_children():
            widget.destroy()
        self.const_entries = []
        if sol is not None:
            x = symbols('x')
            c_syms = sorted([s for s in sol.free_symbols if str(s).startswith('C')], key=lambda s: str(s))
            for cs in c_syms:
                row = ctk.CTkFrame(self.consts_frame, fg_color="transparent")
                row.pack(side="left", padx=(2,8))
                ctk.CTkLabel(row, text=f"{str(cs)}=").pack(side="left")
                e = ctk.CTkEntry(row, width=55)
                e.pack(side="left")
                self.const_entries.append((cs, e))
        if self.ode_order == 1:
            self.dirfield_btn.configure(state="normal")
        else:
            self.dirfield_btn.configure(state="disabled")

        self.log_textbox.configure(state="normal")
        self.log_textbox.delete("1.0", ctk.END)
        self.log_textbox.insert("1.0", self.log_text or LANG[self.lang]["no_log"])
        self.log_textbox.configure(state="disabled")

        # Math image render (always in main thread)
        if sol is not None and ode_expr is not None and rhs is not None:
            import matplotlib.pyplot as plt
            from sympy.printing.latex import latex

            eq_pretty = prettify_floats(Eq(ode_expr, rhs))
            sol_pretty = prettify_floats(sol)

            eq_tex = r"$" + latex(eq_pretty) + r"$"
            ans_tex = r"$" + latex(sol_pretty) + r"$"
            maxlen = max(len(eq_tex), len(ans_tex))
            if maxlen < 70:
                font_size = 18
            elif maxlen < 110:
                font_size = 14
            elif maxlen < 170:
                font_size = 10
            elif maxlen < 300:
                font_size = 8
            else:
                font_size = 7

            fig, ax = plt.subplots(figsize=(7,1.3))
            ax.axis('off')
            ax.text(0.5, 0.78, eq_tex, fontsize=font_size, ha='center', va='center', fontweight='bold')
            ax.text(0.5, 0.38, ans_tex, fontsize=font_size, ha='center', va='center', color='#1858af')
            buf = BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.15, dpi=160, transparent=True)
            plt.close(fig)
            buf.seek(0)

            pil_img = Image.open(buf)
            self.answer_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=pil_img.size)
            self.all_images.append(self.answer_img)  # Keep a reference!
            self.answer_img_label.configure(image=self.answer_img, text="")
        else:
            self.answer_img_label.configure(image=None, text="")
            self.answer_img = None

    def show_input_page(self):
        self.result_page.pack_forget()
        self.help_page.pack_forget()
        self.input_page.pack(fill="both", expand=True)
        self.status_label.configure(text="")
        self.set_enable_state(True)
        if getattr(self, "plot_canvas", None):
            self.plot_canvas.get_tk_widget().destroy()
            self.plot_canvas = None
        if getattr(self, "dirfield_canvas", None):
            self.dirfield_canvas.get_tk_widget().destroy()
            self.dirfield_canvas = None

    def show_result_section(self, section):
        self.answer_frame.pack_forget()
        self.plotter_frame.pack_forget()
        self.dirfield_frame.pack_forget()
        self.log_frame.pack_forget()
        if section == "answer":
            self.answer_frame.pack(fill="both", expand=True, pady=(15,5))
            if hasattr(self, "answer_img_label") and getattr(self, "answer_img", None) is not None:
                self.answer_img_label.configure(image=self.answer_img, text="")
        elif section == "plotter":
            self.plotter_frame.pack(fill="both", expand=True, pady=(15,5))
        elif section == "dirfield":
            self.dirfield_frame.pack(fill="both", expand=True, pady=(15,5))
        elif section == "log":
            self.log_frame.pack(fill="both", expand=True, pady=(15,5))

    def create_coefficient_entries(self):
        for widget in self.coeff_scrollable.winfo_children():
            widget.destroy()
        self.coeff_entries = []
        try:
            n = int(self.entry_n.get())
            if n < 1:
                raise ValueError
        except ValueError:
            self.status_label.configure(text=LANG[self.lang]["status_order_err"], text_color="red")
            return
        for i in range(n, 0, -1):
            row_frame = ctk.CTkFrame(self.coeff_scrollable, fg_color="transparent")
            row_frame.pack(anchor="w", fill="x", pady=1)
            label = ctk.CTkLabel(row_frame, text=f"{LANG[self.lang]['order_label']} y^({i}):", width=160)
            label.pack(side="left", anchor="w", padx=2)
            entry = ctk.CTkEntry(row_frame, width=90)
            entry.pack(side="left", anchor="w", padx=2)
            self.coeff_entries.append(entry)
        row_frame = ctk.CTkFrame(self.coeff_scrollable, fg_color="transparent")
        row_frame.pack(anchor="w", fill="x", pady=1)
        label = ctk.CTkLabel(row_frame, text=f"{LANG[self.lang]['order_label']} y:", width=160)
        label.pack(side="left", anchor="w", padx=2)
        entry = ctk.CTkEntry(row_frame, width=90)
        entry.pack(side="left", anchor="w", padx=2)
        self.coeff_entries.append(entry)
        self.entry_n.configure(state="disabled")
        self.button_set_order.configure(state="disabled")

    def restart_program(self):
        import gc
        self.entry_n.configure(state="normal")
        self.entry_n.delete(0, ctk.END)
        self.button_set_order.configure(state="normal")
        for widget in self.coeff_scrollable.winfo_children():
            widget.destroy()
        self.coeff_entries = []
        self.entry_fx.delete(0, ctk.END)
        self.status_label.configure(text="", text_color="white")
        self.button_solve.configure(state="normal")
        self.button_stop.configure(state="disabled")
        self.input_page.pack(fill="both", expand=True)
        self.result_page.pack_forget()
        self.help_page.pack_forget()
        self.log_text = ""
        self.answer_img_label.configure(image=None, text="")
        self.answer_text.delete("1.0", ctk.END)
        self.ode_order = 1
        self._stop_solve.clear()
        self.all_images.clear()
        gc.collect()

    def set_enable_state(self, enabled):
        state = "normal" if enabled else "disabled"
        widgets = [self.entry_fx, self.button_solve]
        for w in widgets:
            w.configure(state=state)
        for e in self.coeff_entries:
            e.configure(state=state)
        if enabled:
            self.button_stop.configure(state="disabled")
        else:
            self.button_stop.configure(state="normal")

    def stop_solve(self):
        self._stop_solve.set()
        self.status_label.configure(text=LANG[self.lang]["status_stopped"], text_color="red")
        self.set_enable_state(True)
        self.button_stop.configure(state="disabled")

    def start_solve(self):
        self.status_label.configure(text=LANG[self.lang]["status_solving"], text_color="white")
        self.set_enable_state(False)
        self.button_stop.configure(state="normal")
        self._stop_solve.clear()
        self.solving_thread = threading.Thread(target=self.solve_ode_thread, daemon=True)
        self.solving_thread.start()

    def solve_ode_thread(self):
        x = symbols('x')
        y = Function('y')
        try:
            n = int(self.entry_n.get())
        except ValueError:
            self.after(0, lambda: self.show_solve_error(LANG[self.lang]["status_order_int_err"]))
            return
        try:
            coeffs = [float(e.get()) for e in self.coeff_entries]
        except Exception:
            self.after(0, lambda: self.show_solve_error(LANG[self.lang]["status_coeff_err"]))
            return
        fx_input = self.entry_fx.get()
        if fx_input.strip() == "0":
            f_expr = 0
        else:
            try:
                f_expr = sympify(fx_input)
            except Exception:
                self.after(0, lambda: self.show_solve_error(LANG[self.lang]["status_fx_err"]))
                return

        ode_expr = 0
        order = n
        for coef in coeffs:
            if order == 0:
                ode_expr += coef * y(x)
            else:
                ode_expr += coef * Derivative(y(x), x, order)
            order -= 1
        eq = Eq(ode_expr, f_expr)

        from sympy.printing.str import sstr
        log_text = ""
        log_text += f"Equation: {sstr(eq)}\n"
        log_text += "Start solving...\n"

        try:
            sol = dsolve(eq, y(x))
            log_text += f"Answer: {sstr(sol)}\n"
            log_text += "Done."
            result_str = pretty(sol)
            if self._stop_solve.is_set():
                return
            self.after(0, lambda: self.on_solve_done(result_str, n, sol, f_expr, ode_expr, log_text))
        except Exception as e:
            log_text += f"Error: {e}\n"
            self.after(0, lambda: self.show_solve_error(LANG[self.lang]["status_dirfield_failed"]))
            self.after(0, lambda: self.button_stop.configure(state="disabled"))
    
    def on_solve_done(self, result_str, n, sol, f_expr, ode_expr, log_text):
        if self._stop_solve.is_set():
            return
        self.log_text = log_text
        self.show_result_page(result_str, n, sol, f_expr, ode_expr)
        self.button_stop.configure(state="disabled")
    
    def show_solve_error(self, msg):
        self.after(100, lambda: [
            self.status_label.configure(text=msg, text_color="red"),
            self.set_enable_state(True)
        ])

    def plot_solution(self):
        if not hasattr(self, "solution_expr") or self.solution_expr is None:
            return
        x = symbols('x')
        try:
            x_start = float(self.range_x0_entry.get())
            x_end = float(self.range_x1_entry.get())
            if x_end <= x_start:
                raise Exception()
        except:
            self.plotter_label.configure(text=LANG[self.lang]["status_plot_range_err"], text_color="red")
            return
        expr = self.solution_expr.rhs
        for cs, entry in self.const_entries:
            val = entry.get()
            try:
                val = float(val)
            except:
                self.plotter_label.configure(text=LANG[self.lang]["status_plot_const"].format(const=cs), text_color="red")
                return
            expr = expr.subs(cs, val)
        try:
            func = lambdify(x, expr, modules=['numpy'])
            xs = np.linspace(x_start, x_end, 400)
            ys = func(xs)
        except Exception:
            self.plotter_label.configure(text=LANG[self.lang]["status_plot_failed"], text_color="red")
            return
        fig, ax = plt.subplots(figsize=(5.5, 3.5), dpi=100)
        ax.plot(xs, ys, label="y(x)")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(LANG[self.lang]["plot_solution"])
        ax.grid(True)
        ax.legend()
        if self.plot_canvas:
            self.plot_canvas.get_tk_widget().destroy()
        self.plot_canvas = FigureCanvasTkAgg(fig, master=self.plot_canvas_frame)
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)
        self.plotter_label.configure(text=LANG[self.lang]["plot_solution"], text_color="white")

    def plot_directional_field(self):
        if self.ode_order != 1:
            return
        x = symbols('x')
        yfunc = Function('y')
        y_sym = symbols('y')
        try:
            x_start = float(self.dirfield_x0_entry.get())
            x_end = float(self.dirfield_x1_entry.get())
            if x_end <= x_start:
                raise Exception()
        except Exception:
            self.dirfield_label.configure(text=LANG[self.lang]["status_dirfield_range_err"], text_color="red")
            return
        try:
            deriv = Derivative(yfunc(x), x)
            from sympy import solve
            solved = solve(self.ode_expr - self.ode_rhs, deriv)
            if solved:
                rhs_expr = solved[0]
                rhs_expr = rhs_expr.subs(yfunc(x), y_sym)
            else:
                rhs_expr = self.ode_rhs
                if hasattr(rhs_expr, 'subs'):
                    rhs_expr = rhs_expr.subs(yfunc(x), y_sym)
        except Exception as e:
            self.dirfield_label.configure(text=LANG[self.lang]["status_dirfield_parse"], text_color="red")
            return
        try:
            f = lambdify((x, y_sym), rhs_expr, modules=["numpy"])
            x_vals = np.linspace(x_start, x_end, 25)
            y_vals = np.linspace(-5, 5, 25)
            X, Y = np.meshgrid(x_vals, y_vals)
            U = np.ones_like(X)
            V = f(X, Y)
            L = np.sqrt(U**2 + V**2)
            fig, ax = plt.subplots(figsize=(5.5, 3.5), dpi=100)
            ax.quiver(X, Y, U/L, V/L, angles='xy')
            ax.set_title(LANG[self.lang]["dirfield"])
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.grid(True)
            if self.dirfield_canvas:
                self.dirfield_canvas.get_tk_widget().destroy()
            self.dirfield_canvas = FigureCanvasTkAgg(fig, master=self.dirfield_canvas_frame)
            self.dirfield_canvas.draw()
            self.dirfield_canvas.get_tk_widget().pack(fill="both", expand=True)
            plt.close(fig)
            self.dirfield_label.configure(text=LANG[self.lang]["dirfield"], text_color="white")
        except Exception as e:
            self.dirfield_label.configure(text=LANG[self.lang]["status_dirfield_failed"], text_color="red")

if __name__ == "__main__":
    app = ODEGui()
    app.mainloop()
