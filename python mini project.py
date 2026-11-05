import os
import re
import threading
import pickle
import tkinter as tk
from tkinter import ttk, messagebox

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

MODEL_FILE = "spam_model.pkl"

# ----------------------------------------------------------------------
# 1. NATURAL LANGUAGE PREPROCESSING
# ----------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Standardizes input: removes URLs, HTML tags, special chars, and lowercases."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ----------------------------------------------------------------------
# 2. BENCHMARK DATASET GENERATOR
# ----------------------------------------------------------------------
def get_benchmark_dataset() -> pd.DataFrame:
    """Generates an initial training corpus for immediate cold-start training."""
    ham_samples = [
        "Hi, please check the attached project report draft and let me know your thoughts.",
        "Are we meeting in the lab this afternoon for the operating systems lab assignment?",
        "Your package has been dispatched and is scheduled for delivery tomorrow afternoon.",
        "Reminder: The seminar on distributed systems starts at 10 AM in Hall 3.",
        "Thanks for sharing the lecture notes, I will review them before the quiz.",
        "Can you push your latest commit to GitHub so we can run the test suite?",
        "Please find the meeting minutes from this morning's team coordination sync.",
        "Don't forget to submit your lab record before the deadline on Friday.",
        "Let's grab lunch after class near the university cafeteria.",
        "Attached is the revised project proposal with the requested methodology changes."
    ] * 35

    spam_samples = [
        "URGENT: Your account has been compromised! Click here immediately to verify credentials.",
        "Congratulations! You won a $1,000 cash prize. Claim your free gift card now!",
        "Double your money in 48 hours with our automated trading system. Sign up free!",
        "Final notice: You have an unclaimed lottery winning. Reply with your bank details.",
        "Exclusive limited-time offer: 85% discount on all designer watches today only!",
        "Earn $500 an hour working from home without any upfront fee. Click this link.",
        "You have been pre-selected for an instant low-interest loan. Apply in 60 seconds.",
        "Claim your complimentary iPhone 15 by filling out this 2-minute promotional survey.",
        "Warning: Immediate verification required to prevent permanent suspension of your inbox.",
        "Free crypto airdrop! Connect your wallet now to claim 500 bonus tokens."
    ] * 35

    df_ham = pd.DataFrame({"label": "ham", "text": ham_samples})
    df_spam = pd.DataFrame({"label": "spam", "text": spam_samples})
    df = pd.concat([df_ham, df_spam], ignore_index=True)
    return df.sample(frac=1.0, random_state=42).reset_index(drop=True)

# ----------------------------------------------------------------------
# 3. APPLICATION GUI & INFERENCE PIPELINE
# ----------------------------------------------------------------------
class SpamDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MailShield AI - Spam Detection System")
        self.root.geometry("780x740")
        self.root.minsize(650, 600)
        self.root.configure(bg="#f1f3f5")

        self.vectorizer = None
        self.model = None

        self._init_ui()
        self._load_saved_model()

    def _init_ui(self):
        style = ttk.Style()
        style.theme_use("clam")

        # Header Banner
        header = tk.Frame(self.root, bg="#1a202c", pady=14)
        header.pack(fill=tk.X)
        tk.Label(
            header,
            text="MailShield AI: Spam Email Classifier",
            font=("Helvetica", 14, "bold"),
            fg="#ffffff",
            bg="#1a202c"
        ).pack()

        # Training Frame
        train_frame = ttk.LabelFrame(self.root, text="Model Controls & Status", padding=12)
        train_frame.pack(fill=tk.X, padx=14, pady=10)

        self.btn_train = ttk.Button(train_frame, text="Train Model (Benchmark Corpus)", command=self.train_model_async)
        self.btn_train.grid(row=0, column=0, padx=5, sticky=tk.W)

        self.train_pbar = ttk.Progressbar(train_frame, mode="indeterminate", length=180)
        self.train_pbar.grid(row=0, column=1, padx=10, sticky=tk.W)

        self.lbl_train_status = tk.Label(train_frame, text="Checking model state...", font=("Helvetica", 9), fg="#495057")
        self.lbl_train_status.grid(row=0, column=2, padx=10, sticky=tk.W)

        self.lbl_metrics = tk.Label(train_frame, text="", font=("Consolas", 9), fg="#212529", justify=tk.LEFT)
        self.lbl_metrics.grid(row=1, column=0, columnspan=3, pady=(8, 0), sticky=tk.W)

        # Classification Testing Frame
        infer_frame = ttk.LabelFrame(self.root, text="Live Message Inspector", padding=12)
        infer_frame.pack(fill=tk.BOTH, expand=True, padx=14, pady=5)

        tk.Label(infer_frame, text="Type or paste email text to evaluate:", font=("Helvetica", 9)).pack(anchor=tk.W, pady=(0, 4))

        self.txt_input = tk.Text(infer_frame, height=7, font=("Helvetica", 10), relief=tk.SOLID, bd=1)
        self.txt_input.pack(fill=tk.X, pady=4)

        btn_row = tk.Frame(infer_frame)
        btn_row.pack(fill=tk.X, pady=6)

        ttk.Button(btn_row, text="Analyze Email", command=self.classify_text).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_row, text="Clear", command=self._clear_input).pack(side=tk.LEFT)

        # Results Display Card
        results_card = ttk.LabelFrame(infer_frame, text="Classification Result", padding=10)
        results_card.pack(fill=tk.BOTH, expand=True, pady=8)

        self.lbl_verdict = tk.Label(
            results_card,
            text="VERDICT: AWAITING INPUT",
            font=("Helvetica", 12, "bold"),
            fg="#6c757d"
        )
        self.lbl_verdict.pack(anchor=tk.W, pady=(0, 8))

        # Ham Confidence Bar
        tk.Label(results_card, text="Ham (Legitimate) Probability:", font=("Helvetica", 9)).pack(anchor=tk.W)
        self.pbar_ham = ttk.Progressbar(results_card, orient=tk.HORIZONTAL, mode="determinate")
        self.pbar_ham.pack(fill=tk.X, pady=2)
        self.lbl_ham = tk.Label(results_card, text="0.0%", font=("Helvetica", 8), fg="#495057")
        self.lbl_ham.pack(anchor=tk.W)

        # Spam Confidence Bar
        tk.Label(results_card, text="Spam Probability:", font=("Helvetica", 9), pady=4).pack(anchor=tk.W)
        self.pbar_spam = ttk.Progressbar(results_card, orient=tk.HORIZONTAL, mode="determinate")
        self.pbar_spam.pack(fill=tk.X, pady=2)
        self.lbl_spam = tk.Label(results_card, text="0.0%", font=("Helvetica", 8), fg="#495057")
        self.lbl_spam.pack(anchor=tk.W)

    def _clear_input(self):
        self.txt_input.delete("1.0", tk.END)
        self.lbl_verdict.config(text="VERDICT: AWAITING INPUT", fg="#6c757d")
        self.pbar_ham["value"] = 0
        self.pbar_spam["value"] = 0
        self.lbl_ham.config(text="0.0%")
        self.lbl_spam.config(text="0.0%")

    def _load_saved_model(self):
        """Loads previously saved model artifacts from disk if available."""
        if os.path.exists(MODEL_FILE):
            try:
                with open(MODEL_FILE, "rb") as f:
                    data = pickle.load(f)
                    self.model = data["model"]
                    self.vectorizer = data["vectorizer"]
                self.lbl_train_status.config(text="Trained model loaded from disk.", fg="#2b8a3e")
                return
            except Exception:
                pass
        self.lbl_train_status.config(text="No trained model found. Click 'Train Model' to start.", fg="#c92a2a")

    def train_model_async(self):
        """Trains the TF-IDF + MultinomialNB pipeline in a worker thread."""
        self.btn_train.config(state=tk.DISABLED)
        self.train_pbar.start(10)
        self.lbl_train_status.config(text="Training model on background thread...", fg="#1864ab")

        def worker():
            try:
                df = get_benchmark_dataset()
                cleaned_corpus = [clean_text(t) for t in df["text"]]
                y = (df["label"] == "spam").astype(int).values

                X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                    cleaned_corpus, y, test_size=0.2, random_state=42, stratify=y
                )

                vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=2500, stop_words="english")
                X_train_vec = vectorizer.fit_transform(X_train_raw)
                X_test_vec = vectorizer.transform(X_test_raw)

                model = MultinomialNB(alpha=1.0)
                model.fit(X_train_vec, y_train)

                y_pred = model.predict(X_test_vec)
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred)
                rec = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)

                with open(MODEL_FILE, "wb") as f:
                    pickle.dump({"model": model, "vectorizer": vectorizer}, f)

                self.model = model
                self.vectorizer = vectorizer

                def update_success():
                    self.train_pbar.stop()
                    self.btn_train.config(state=tk.NORMAL)
                    self.lbl_train_status.config(text="Model converged and ready for testing.", fg="#2b8a3e")
                    self.lbl_metrics.config(
                        text=f"Validation Metrics -> Accuracy: {acc*100:.1f}% | Precision: {prec*100:.1f}% | Recall: {rec*100:.1f}% | F1: {f1*100:.1f}%"
                    )
                    messagebox.showinfo("Training Complete", f"Trained successfully!\nAccuracy: {acc*100:.1f}%\nPrecision: {prec*100:.1f}%")

                self.root.after(0, update_success)
            except Exception as e:
                def update_err():
                    self.train_pbar.stop()
                    self.btn_train.config(state=tk.NORMAL)
                    self.lbl_train_status.config(text="Error occurred during training.", fg="#c92a2a")
                    messagebox.showerror("Error", str(e))
                self.root.after(0, update_err)

        threading.Thread(target=worker, daemon=True).start()

    def classify_text(self):
        """Processes user input and displays predicted classification probabilities."""
        if not self.model or not self.vectorizer:
            messagebox.showwarning("Model Missing", "Please train the model first by clicking 'Train Model'.")
            return

        raw_text = self.txt_input.get("1.0", tk.END).strip()
        if not raw_text:
            messagebox.showwarning("Empty Message", "Please paste or type an email message to analyze.")
            return

        cleaned = clean_text(raw_text)
        features = self.vectorizer.transform([cleaned])
        probs = self.model.predict_proba(features)[0]

        ham_prob = probs[0] * 100
        spam_prob = probs[1] * 100

        self.pbar_ham["value"] = ham_prob
        self.pbar_spam["value"] = spam_prob
        self.lbl_ham.config(text=f"{ham_prob:.2f}%")
        self.lbl_spam.config(text=f"{spam_prob:.2f}%")

        if spam_prob >= 50.0:
            self.lbl_verdict.config(text=f"VERDICT: SPAM DETECTED ({spam_prob:.1f}% confidence)", fg="#c92a2a")
        else:
            self.lbl_verdict.config(text=f"VERDICT: HAM / LEGITIMATE ({ham_prob:.1f}% confidence)", fg="#2b8a3e")

# ----------------------------------------------------------------------
# 4. ENTRY POINT
# ----------------------------------------------------------------------
if __name__ == "__main__":
    app_root = tk.Tk()
    app = SpamDetectorApp(app_root)
    app_root.mainloop()