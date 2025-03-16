import sys
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QLineEdit, QPushButton
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from styles import TEXT_AREA_STYLE, INPUT_LINE_STYLE, BUTTON_STYLE_ACTIVE, BUTTON_STYLE_INACTIVE

print("""
  __  __          _____  ______      ______     __      _____ ______ _      _____ _______ ____  _____  
 |  \/  |   /\   |  __ \|  ____|    |  _ \ \   / /     / ____|  ____| |    |_   _|__   __|___ \|  __ \ 
 | \  / |  /  \  | |  | | |__       | |_) \ \_/ /     | (___ | |__  | |      | |    | |    __) | |__) |
 | |\/| | / /\ \ | |  | |  __|      |  _ < \   /       \___ \|  __| | |      | |    | |   |__ <|  _  / 
 | |  | |/ ____ \| |__| | |____     | |_) | | |        ____) | |____| |____ _| |_   | |   ___) | | \ \ 
 |_|  |_/_/    \_\_____/|______|    |____/  |_|       |_____/|______|______|_____|  |_|  |____/|_|  \_\ 
      """)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    print(f"Используется видеокарта: {torch.cuda.get_device_name(0)}")
else:
    print("Видеокарта не найдена, используется CPU.")

model_name = "MTS-AI/cotype-nano-1.5b"
cache_dir = "MTS-AI/temp"

if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.abspath(".")

model_path = os.path.join(base_path, "MTS-AI", "cotype-nano-1.5b")
os.makedirs(model_path, exist_ok=True)

def load_model():
    start_time = time.time()
    tokenizer, model = None, None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_path)
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=model_path).to(device)
    except Exception as e:
        print(f"Ошибка при загрузке: {e}")
    if not tokenizer or not model:
        print("Не удалось загрузить модель или токенизатор.")
    else:
        print(f"Загрузка завершена за {time.time() - start_time:.2f} секунд.")
    return tokenizer, model

tokenizer, model = load_model()
dialog_history = ""

def generate_text(prompt, short_answer=False):
    global dialog_history
    if short_answer:
        dialog_history = ""

    prompt_with_history = dialog_history + f"<|user|>{prompt}\n<|assistant|>"

    inputs = tokenizer(prompt_with_history, return_tensors="pt", padding=True, truncation=True)

    inputs['input_ids'] = torch.cat([torch.tensor([[tokenizer.eos_token_id]], device=device), inputs['input_ids']], dim=1)
    inputs['attention_mask'] = torch.cat([torch.tensor([[1]], device=device), inputs['attention_mask']], dim=1)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    try:
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=18000,
            no_repeat_ngram_size=0,
            temperature=1,
            top_p=0.9,
            top_k=500,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if generated_text.lower().startswith(prompt_with_history.lower()):
            generated_text = generated_text[len(prompt_with_history):].strip()

        dialog_history += f"<|system|>Ты — AI-ассистент. Отвечай только на заданный вопрос без дополнительных рассуждений.\n<|user|>{prompt}\n<|assistant|>{generated_text}\n"
        
        return generated_text
    except Exception as e:
        print(f"Ошибка генерации: {e}")
        return "Ошибка при генерации текста."

class ChatApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Chat")
        self.setGeometry(300, 300, 600, 400)
        self.setStyleSheet("background-color: #181818; font-family: 'Roboto', sans-serif;")

        self.layout = QVBoxLayout()

        self.text_area = QTextEdit(self)
        self.text_area.setReadOnly(True)
        self.text_area.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.text_area.setStyleSheet(TEXT_AREA_STYLE)
        self.layout.addWidget(self.text_area)

        self.input_line = QLineEdit(self)
        self.input_line.setStyleSheet(INPUT_LINE_STYLE)
        self.layout.addWidget(self.input_line)

        self.submit_button = QPushButton("Отправить", self)
        self.submit_button.setStyleSheet(BUTTON_STYLE_ACTIVE)
        self.submit_button.clicked.connect(self.on_submit)
        self.layout.addWidget(self.submit_button)

        self.setLayout(self.layout)
        self.thread = None
        self.generated_text = ""  
        self.current_text = ""  

    def update_button_style(self, active=True):
        self.submit_button.setStyleSheet(BUTTON_STYLE_ACTIVE if active else BUTTON_STYLE_INACTIVE)

    def on_submit(self):
        user_input = self.input_line.text()
        if user_input.strip():
            self.text_area.append(f"User: {user_input}")
            self.input_line.clear()
            self.input_line.setDisabled(True)
            self.update_button_style(active=False)

            short_answer = len(user_input.split()) == 1

            if self.thread and self.thread.isRunning():
                self.thread.quit()
                self.thread.wait()

            self.thread = QThread()
            self.worker = GenerateTextWorker(user_input, short_answer)
            self.worker.moveToThread(self.thread)
            self.worker.finished.connect(self.on_text_generated)
            self.thread.started.connect(self.worker.run)
            self.thread.start()

    def on_text_generated(self, response):
        self.generated_text = response
        self.current_text = f"AI: {self.generated_text}"
        self.text_area.append(self.current_text)
        self.input_line.setDisabled(False)
        self.update_button_style(active=True)

class GenerateTextWorker(QThread):
    finished = pyqtSignal(str)

    def __init__(self, prompt, short_answer):
        super().__init__()
        self.prompt = prompt
        self.short_answer = short_answer

    def run(self):
        response = generate_text(self.prompt, self.short_answer)
        self.finished.emit(response)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChatApp()
    window.show()
    sys.exit(app.exec())
