import os
import time
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

from PIL import Image, ImageTk

import tensorflow as tf
from tensorflow import keras

import requests

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "models/EfficientNet_sasl_letters_model.keras"   # same folder
IMG_SIZE = 224

# ROI (same idea as your data-collection script)
ROI_TOP = 10

# preprocessing params (match your script)
MIN_VALUE = 70
BLUR_KERNEL = (5, 5)
BLUR_SIGMA = 2
ADAPT_BLOCK = 11
ADAPT_C = 2

# Auto-accept logic
AUTO_ACCEPT_THRESHOLD = 0.85
AUTO_ACCEPT_SECONDS = 3.0

# Model class order (index 0 is SPACE)
CLASS_NAMES = ["SPACE"] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# -----------------------------
# TRANSLATION (LibreTranslate API)
# -----------------------------
# You can change these without touching code by setting environment variables:
#   set LIBRETRANSLATE_URL=https://libretranslate.com
#   set LIBRETRANSLATE_API_KEY=your_key_here
LIBRETRANSLATE_URL = os.environ.get("LIBRETRANSLATE_URL", "https://libretranslate.com")
LIBRETRANSLATE_API_KEY = os.environ.get("LIBRETRANSLATE_API_KEY", "").strip()

# Language dropdown -> ISO code (LibreTranslate expects codes)
# NOTE: “All African languages” is huge + not all APIs support all.
# This list includes many common African languages + the ones you requested.
AFRICAN_LANG_MAP = {
    "Afrikaans": "af",
    "Amharic": "am",
    "Arabic": "ar",
    "Bambara": "bm",
    "Chichewa (Chewa)": "ny",
    "Ewe": "ee",
    "Fula (Fulah)": "ff",
    "Hausa": "ha",
    "Igbo": "ig",
    "Kinyarwanda": "rw",
    "Kirundi": "rn",
    "Lingala": "ln",
    "Malagasy": "mg",
    "Oromo": "om",
    "Shona": "sn",
    "Somali": "so",
    "Sesotho": "st",
    "Setswana (Tswana)": "tn",
    "Tigrinya": "ti",
    "Twi (Akan)": "ak",
    "Wolof": "wo",
    "Xhosa": "xh",
    "Yoruba": "yo",
    "Zulu": "zu",
    "Swahili": "sw",
    "Ndebele": "nr",     # may not be supported on some LT servers
    "Venda": "ve",       # may not be supported on some LT servers
    "Xitsonga (Tsonga)": "ts",  # may not be supported on some LT servers
}

# -----------------------------
# Helpers
# -----------------------------
def preprocess_roi(roi_bgr: np.ndarray) -> np.ndarray:
    """Match your training pipeline: gray -> blur -> adaptive -> Otsu -> resize to 224"""
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, BLUR_KERNEL, BLUR_SIGMA)

    th3 = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        ADAPT_BLOCK,
        ADAPT_C
    )

    _, out = cv2.threshold(th3, MIN_VALUE, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    out = cv2.resize(out, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    return out

def to_model_input(processed_224: np.ndarray) -> np.ndarray:
    """
    EfficientNet expects 3 channels.
    Convert 1-channel binary -> RGB (3-channel) then preprocess_input.
    """
    rgb = cv2.cvtColor(processed_224, cv2.COLOR_GRAY2RGB)  # (224,224,3)
    x = rgb.astype(np.float32)
    x = np.expand_dims(x, axis=0)  # (1,224,224,3)
    x = keras.applications.efficientnet.preprocess_input(x)
    return x

def cv_to_tk_image(bgr: np.ndarray, target_w: int, target_h: int) -> ImageTk.PhotoImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb).resize((target_w, target_h), Image.Resampling.BILINEAR)
    return ImageTk.PhotoImage(img)

def gray_to_tk_image(gray: np.ndarray, target_w: int, target_h: int) -> ImageTk.PhotoImage:
    img = Image.fromarray(gray).resize((target_w, target_h), Image.Resampling.NEAREST)
    return ImageTk.PhotoImage(img.convert("L"))

def libretranslate_translate(text: str, target_code: str) -> str:
    """
    Calls LibreTranslate API:
      POST /translate  {q, source, target, format, api_key}
    """
    url = LIBRETRANSLATE_URL.rstrip("/") + "/translate"
    payload = {
        "q": text,
        "source": "auto",
        "target": target_code,
        "format": "text",
    }
    if LIBRETRANSLATE_API_KEY:
        payload["api_key"] = LIBRETRANSLATE_API_KEY

    r = requests.post(url, data=payload, timeout=20)
    r.raise_for_status()
    data = r.json()
    if "translatedText" not in data:
        raise RuntimeError(f"Unexpected API response: {data}")
    return data["translatedText"]

# -----------------------------
# Main App
# -----------------------------
class SASLApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("SASL Recognition + Spell + Translator")
        self.root.geometry("1100x620")
        self.root.minsize(1100, 620)

        # ---- Load model
        if not os.path.exists(MODEL_PATH):
            messagebox.showerror("Model not found", f"Could not find: {MODEL_PATH}\nPut it in the same folder.")
            raise FileNotFoundError(MODEL_PATH)
        self.model = keras.models.load_model(MODEL_PATH)

        # ---- Webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Webcam error", "Could not open webcam (VideoCapture(0)).\nTry closing other camera apps.")
            raise RuntimeError("Could not open webcam")

        # ---- State
        self.last_pred = None
        self.last_pred_time = time.time()
        self.accepted_cooldown_until = 0.0

        self.current_word = ""
        self.current_sentence = ""

        # ---- Build UI
        self._build_layout()

        # ---- start loop
        self.update_loop()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_layout(self):
        top = ttk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=6)

        bottom = ttk.LabelFrame(self.root, text="Translator")
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=6)

        top.columnconfigure(0, weight=4)
        top.columnconfigure(1, weight=2)
        top.columnconfigure(2, weight=2)
        top.rowconfigure(0, weight=1)

        # LEFT: Webcam frame
        left_frame = ttk.LabelFrame(top, text="Webcam")
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        self.webcam_label = ttk.Label(left_frame)
        self.webcam_label.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # MIDDLE: Suggestions
        mid_frame = ttk.LabelFrame(top, text="Suggestions")
        mid_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 6))

        mid_inner = tk.Frame(mid_frame, bg="#efefef")
        mid_inner.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        tk.Label(mid_inner, text="Suggestions:", bg="#efefef", font=("Arial", 11, "bold")).pack(anchor="w", pady=(0, 8))

        self.sug_btn1 = ttk.Button(mid_inner, text="Suggest 1", command=lambda: self.apply_suggestion(0))
        self.sug_btn2 = ttk.Button(mid_inner, text="Suggest 2", command=lambda: self.apply_suggestion(1))
        self.sug_btn3 = ttk.Button(mid_inner, text="Suggest 3", command=lambda: self.apply_suggestion(2))

        self.sug_btn1.pack(pady=4, fill=tk.X)
        self.sug_btn2.pack(pady=4, fill=tk.X)
        self.sug_btn3.pack(pady=4, fill=tk.X)

        self.suggestions = ["", "", ""]
        self.update_suggestion_buttons()

        # RIGHT: Output + Controls + ROI preview (ROI is BELOW buttons)
        right_frame = ttk.LabelFrame(top, text="Output + Controls")
        right_frame.grid(row=0, column=2, sticky="nsew")

        right_inner = tk.Frame(right_frame, bg="#f7f7ff")
        right_inner.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.pred_var = tk.StringVar(value="Prediction: - (0.00)")
        self.word_var = tk.StringVar(value="Word: ")
        self.sent_var = tk.StringVar(value="Sentence: ")

        tk.Label(right_inner, textvariable=self.pred_var, bg="#f7f7ff", font=("Arial", 11, "bold")).pack(anchor="w")
        tk.Label(right_inner, textvariable=self.word_var, bg="#f7f7ff", font=("Arial", 11)).pack(anchor="w", pady=(6, 0))
        tk.Label(right_inner, textvariable=self.sent_var, bg="#f7f7ff", font=("Arial", 11)).pack(anchor="w", pady=(2, 8))

        btn_frame = tk.Frame(right_inner, bg="#f7f7ff")
        btn_frame.pack(fill=tk.X)

        ttk.Button(btn_frame, text="Add Letter", command=self.manual_add_letter).pack(fill=tk.X, pady=3)
        ttk.Button(btn_frame, text="Space", command=self.manual_space).pack(fill=tk.X, pady=3)
        ttk.Button(btn_frame, text="Backspace", command=self.backspace).pack(fill=tk.X, pady=3)
        ttk.Button(btn_frame, text="Clear", command=self.clear_all).pack(fill=tk.X, pady=3)

        roi_box = ttk.LabelFrame(right_inner, text="ROI (Processed)")
        roi_box.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.roi_label = ttk.Label(roi_box)
        self.roi_label.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Keyboard bindings
        self.root.bind("<space>", lambda e: self.manual_space())
        self.root.bind("<BackSpace>", lambda e: self.backspace())
        self.root.bind(".", lambda e: self.finish_sentence())

        # TRANSLATOR bottom (make sure all visible)
        bottom.columnconfigure(0, weight=3)
        bottom.columnconfigure(1, weight=1)
        bottom.columnconfigure(2, weight=1)
        bottom.columnconfigure(3, weight=1)
        bottom.columnconfigure(4, weight=3)
        bottom.rowconfigure(0, weight=1)

        self.input_text = tk.Text(bottom, height=2, wrap="word")
        self.input_text.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)

        self.lang_var = tk.StringVar(value="Shona")
        self.lang_combo = ttk.Combobox(
            bottom,
            textvariable=self.lang_var,
            values=sorted(AFRICAN_LANG_MAP.keys()),
            state="readonly",
            width=16
        )
        self.lang_combo.grid(row=0, column=1, sticky="ew", padx=6, pady=6)

        # Make translate button label ALWAYS visible
        self.translate_btn = ttk.Button(bottom, text="Translate", command=self.translate_sentence)
        self.translate_btn.grid(row=0, column=2, sticky="ew", padx=6, pady=6)
        self.translate_btn.configure(width=12)

        self.clear_btn = ttk.Button(bottom, text="Clear", command=self.clear_translation)
        self.clear_btn.grid(row=0, column=3, sticky="ew", padx=6, pady=6)
        self.clear_btn.configure(width=10)

        self.output_text = tk.Text(bottom, height=2, wrap="word")
        self.output_text.grid(row=0, column=4, sticky="nsew", padx=6, pady=6)

    # -----------------------------
    # Suggestions / Text state
    # -----------------------------
    def update_suggestion_buttons(self):
        t1 = self.suggestions[0] if self.suggestions[0] else "Suggest 1"
        t2 = self.suggestions[1] if self.suggestions[1] else "Suggest 2"
        t3 = self.suggestions[2] if self.suggestions[2] else "Suggest 3"
        self.sug_btn1.configure(text=t1)
        self.sug_btn2.configure(text=t2)
        self.sug_btn3.configure(text=t3)

    def apply_suggestion(self, idx: int):
        sug = self.suggestions[idx]
        if not sug:
            return
        self.current_word = sug
        self.refresh_text()

    def refresh_text(self):
        self.word_var.set(f"Word: {self.current_word}")
        self.sent_var.set(f"Sentence: {self.current_sentence}")

    def manual_add_letter(self):
        if self.last_pred is None or self.last_pred == "SPACE":
            return
        self.current_word += self.last_pred
        self.refresh_text()
        self.update_spell_suggestions()

    def manual_space(self):
        if self.current_word.strip():
            if self.current_sentence and not self.current_sentence.endswith(" "):
                self.current_sentence += " "
            self.current_sentence += self.current_word.strip() + " "
            self.current_word = ""
            self.refresh_text()
            self.update_spell_suggestions()

    def finish_sentence(self):
        if self.current_word.strip():
            self.manual_space()

        self.current_sentence = self.current_sentence.strip()
        if self.current_sentence and not self.current_sentence.endswith("."):
            self.current_sentence += "."

        self.refresh_text()

        # move to translator input
        self.input_text.delete("1.0", tk.END)
        self.input_text.insert(tk.END, self.current_sentence)

        # reset for next sentence
        self.current_sentence = ""
        self.current_word = ""
        self.refresh_text()
        self.update_spell_suggestions()

    def backspace(self):
        if self.current_word:
            self.current_word = self.current_word[:-1]
        else:
            self.current_sentence = self.current_sentence[:-1]
        self.refresh_text()
        self.update_spell_suggestions()

    def clear_all(self):
        self.current_word = ""
        self.current_sentence = ""
        self.refresh_text()
        self.update_spell_suggestions()

    def clear_translation(self):
        self.input_text.delete("1.0", tk.END)
        self.output_text.delete("1.0", tk.END)

    def update_spell_suggestions(self):
        # keep your placeholder (you can replace with pyspellchecker later)
        w = self.current_word.strip().lower()
        if not w:
            self.suggestions = ["", "", ""]
        else:
            self.suggestions = [w, w + "e", w + "s"]
        self.update_suggestion_buttons()

    # -----------------------------
    # TRANSLATION (REAL API CALL)
    # -----------------------------
    def translate_sentence(self):
        text = self.input_text.get("1.0", tk.END).strip()
        if not text:
            return

        lang_name = self.lang_var.get().strip()
        target_code = AFRICAN_LANG_MAP.get(lang_name)

        if not target_code:
            messagebox.showerror("Language not supported", f"No code mapping for: {lang_name}")
            return

        # UI feedback (so user sees it is working)
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, "Translating... please wait.")

        self.root.update_idletasks()

        try:
            translated = libretranslate_translate(text, target_code)
        except requests.exceptions.RequestException as e:
            messagebox.showerror(
                "Translation API Error",
                "Could not reach the translation API.\n\n"
                "Fix options:\n"
                "1) Check your internet\n"
                "2) Try a different endpoint (LIBRETRANSLATE_URL)\n"
                "3) Some public servers require an API key\n\n"
                f"Details: {e}"
            )
            self.output_text.delete("1.0", tk.END)
            return
        except Exception as e:
            messagebox.showerror("Translation Error", str(e))
            self.output_text.delete("1.0", tk.END)
            return

        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, translated)

    # -----------------------------
    # Webcam loop
    # -----------------------------
    def update_loop(self):
        ok, frame = self.cap.read()
        if ok:
            frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]
            x1 = int(0.5 * w)
            y1 = ROI_TOP
            x2 = w - 10
            y2 = int(0.5 * w)

            y2 = min(y2, h - 1)
            x1 = max(x1, 0)
            x2 = min(x2, w - 1)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                processed = preprocess_roi(roi)

                x = to_model_input(processed)
                probs = self.model.predict(x, verbose=0)[0]
                idx = int(np.argmax(probs))
                conf = float(probs[idx])

                pred_label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else str(idx)

                cv2.putText(frame, f"Pred: {pred_label}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
                cv2.putText(frame, f"Conf: {conf*100:.1f}%", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

                self.pred_var.set(f"Prediction: {pred_label} ({conf:.2f})")

                # AUTO-ACCEPT
                now = time.time()
                if now >= self.accepted_cooldown_until:
                    if pred_label == self.last_pred:
                        if conf >= AUTO_ACCEPT_THRESHOLD and (now - self.last_pred_time) >= AUTO_ACCEPT_SECONDS:
                            if pred_label == "SPACE":
                                self.manual_space()
                                accepted_txt = "ACCEPTED: SPACE"
                            else:
                                self.current_word += pred_label
                                self.refresh_text()
                                self.update_spell_suggestions()
                                accepted_txt = f"ACCEPTED: {pred_label}"

                            self.accepted_cooldown_until = now + 1.0
                            self.last_pred_time = now
                            cv2.putText(frame, accepted_txt, (10, 160),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
                    else:
                        self.last_pred = pred_label
                        self.last_pred_time = now

                webcam_tk = cv_to_tk_image(frame, target_w=640, target_h=360)
                self.webcam_label.configure(image=webcam_tk)
                self.webcam_label.image = webcam_tk

                roi_tk = gray_to_tk_image(processed, target_w=220, target_h=220)
                self.roi_label.configure(image=roi_tk)
                self.roi_label.image = roi_tk

        self.root.after(15, self.update_loop)

    def on_close(self):
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    try:
        style = ttk.Style()
        style.theme_use("clam")
    except Exception:
        pass

    app = SASLApp(root)
    root.mainloop()

