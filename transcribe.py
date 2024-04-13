#! python3.7

import sounddevice
import argparse
import os
import numpy as np
import speech_recognition as sr
import whisper
import torch
import threading

from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QLabel
from PyQt5.QtGui import QFont, QFontMetrics
from PyQt5.QtCore import Qt, QTimer, QPoint, QRect, QSize

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", default="medium", help="Model to use",
            choices=["tiny", "base", "small", "medium", "large", "tiny.en", "base.en", "small.en", "medium.en", "large-v3"])
  parser.add_argument("--input", default=None,
            help="Default input source.", type=str)
  parser.add_argument("--language", default=None,
            help="Default language.", type=str)
  parser.add_argument("--font-size", default=30, type=int,
            help="Font size of HUD.")


  parser.add_argument("--energy_threshold", default=1000,
            help="Energy level for mic to detect.", type=int)
  parser.add_argument("--record_timeout", default=1,
            help="How real time the recording is in seconds.", type=float)
  parser.add_argument("--phrase_timeout", default=3,
            help="How much empty space between recordings before we "
               "consider it a new line in the transcription.", type=float)
  args = parser.parse_args()
  return args

class HUD(QMainWindow):
  max_width_percentage = 0.3
  max_lines = 3
  padding = 10
  stupid_qt = 20

  def __init__(self, font_size):
    super().__init__()

    self.text = "Transcription started"

    # Set window attributes
    self.setWindowFlags(Qt.CustomizeWindowHint|Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
    self.setAttribute(Qt.WA_TranslucentBackground)

    # Set window opacity
    self.setWindowOpacity(0.5)

    # Set central widget
    central_widget = QWidget()
    layout = QHBoxLayout(central_widget)

    label = QLabel(self.text)
    label.setAlignment(Qt.AlignLeft|Qt.AlignVCenter)
    label.setStyleSheet("background-color: rgba(0,0,0,0.5); color: white")
    font = QFont()
    font.setPointSize(font_size)  # Set font size to 24
    label.setFont(font)
    label.setWordWrap(True)
    self.label = label

    layout.addWidget(label)

    # Limit window size to 60% of screen width
    screen_width = QApplication.desktop().screenGeometry().width()
    self.max_width = int(self.max_width_percentage * screen_width)
    self.setFixedWidth(self.max_width)

    self.setStyleSheet(f"padding: {self.padding}px;")
    self.setCentralWidget(central_widget)

    # Start updating the text periodically
    self.update_text_timer = QTimer(self)
    self.update_text_timer.timeout.connect(self.update_text)
    self.update_text_timer.start(100)  # Update text every 100 milliseconds

    # Variables for dragging the window
    self.old_pos = None

  def mousePressEvent(self, event):
    if event.button() == Qt.LeftButton:
      self.old_pos = event.globalPos()

  def mouseMoveEvent(self, event):
    if self.old_pos:
      delta = QPoint(event.globalPos() - self.old_pos)
      self.move(self.x() + delta.x(), self.y() + delta.y())
      self.old_pos = event.globalPos()

  def mouseReleaseEvent(self, event):
    if event.button() == Qt.LeftButton:
      self.old_pos = None

  def update_text(self):
    self.label.setText(self.text)

    # Set label maximum height to show only last 3 lines wrapped
    fm = self.label.fontMetrics()
    label_rect = fm.boundingRect(QRect(0, 0, self.max_width, 1000), Qt.TextWordWrap|Qt.AlignLeft|Qt.AlignVCenter, self.text)
    last_3_lines_height = self.max_lines * fm.lineSpacing() + fm.leading()
    max_label_height = min(label_rect.height(), last_3_lines_height) + self.padding*2 + self.stupid_qt
    self.setFixedHeight(max_label_height)
    self.label.resize(QSize(label_rect.width(), label_rect.height()))
    self.label.move(0, max_label_height - label_rect.height())

class Transcriber():
  def __init__(self, args):
    self.args = args
    self.language = args.language
    self.sample_rate = 44100
    # The last time a recording was retrieved from the queue.
    self.compute_device = "cuda" if torch.cuda.is_available() else "cpu"
    self.model_name = args.model
    self.record_timeout = args.record_timeout
    self.phrase_timeout = args.phrase_timeout

    # Thread safe Queue for passing data from the threaded recording callback.
    self.data_queue = Queue()
    self.current_text = ''
    self.transribe_thread = None
    self.stop_event = threading.Event()
    self.hud_window = None

    # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
    self.recorder = sr.Recognizer()
    self.recorder.energy_threshold = args.energy_threshold
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
    self.recorder.dynamic_energy_threshold = False

    self.init_input_device(args)

    # Load / Download model
    self.audio_model = whisper.load_model(self.model_name, device=self.compute_device)

    with self.source:
      self.recorder.adjust_for_ambient_noise(self.source)

    # Cue the user that we're ready to go.
    print("Model loaded.\n")

  def init_input_device(self, args):
    device_index = None
    if args.input is not None:
      for idx, name in enumerate(sr.Microphone.list_microphone_names()):
        if name == args.input:
          device_index = idx
          break

    if device_index is None:
      # Print the list of available audio devices
      print("Available input devices:")
      for idx, name in enumerate(sr.Microphone.list_microphone_names()):
        print(f"{idx}. {name}")

      if args.input is None:
        # Choose the index of the desired input device
        device_index = int(input("Enter the index of the input device you want to use: "))
      else:
        raise RuntimeError(f'Invalid input device "{args.input}"')

    self.source = sr.Microphone(sample_rate=self.sample_rate, device_index=device_index)

  def record_callback(self, _, audio: sr.AudioData) -> None:
    """
    Threaded callback function to receive audio data when recordings finish.
    audio: An AudioData containing the recorded bytes.
    """
    # Grab the raw bytes and push it into the thread safe queue.
    data = audio.get_raw_data()
    self.data_queue.put(data)

  def display_hud(self):
    app = QApplication([])
    self.hud_window = HUD(font_size=self.args.font_size)
    self.hud_window.show()
    app.exec()

  def start_transribe_thread(self):
    if self.transribe_thread is not None:
      raise RuntimeError("Transription thread already running")

    self.transribe_thread = threading.Thread(target=self.listen)
    self.transribe_thread.start()

  def stop_transribe_thread(self):
    if self.transribe_thread is None:
      return

    self.stop_event.set()
    self.transribe_thread.join()

  def listen(self):
    transcription = ['']
    phrase_time = None

    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    self.recorder.listen_in_background(self.source, self.record_callback, phrase_time_limit=self.record_timeout)

    acc_audio_data = torch.zeros((0,), dtype=torch.float32, device=self.compute_device)
    while not self.stop_event.is_set():
      try:
        now = datetime.now()
        # Pull raw recorded audio from the queue.
        phrase_complete = False
        # If enough time has passed between recordings, consider the phrase complete.
        # Clear the current working audio buffer to start over with the new data.
        if phrase_time and now - phrase_time > timedelta(seconds=self.phrase_timeout):
          phrase_complete = True
          acc_audio_data = torch.zeros((0,), dtype=torch.float32, device=self.compute_device)

        self.data_queue.mutex.acquire()
        audio_data = b''.join(self.data_queue.queue)
        self.data_queue.queue.clear()
        self.data_queue.mutex.release()

        if len(audio_data) == 0:
          sleep(0.1)
          continue

        # Convert in-ram buffer to something the model can use directly without needing a temp file.
        # Convert data from 16 bit wide integers to floating point with a width of 32 bits.
        # Clamp the audio stream frequency to a PCM wavelength compatible default of 32768hz max.
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        audio_torch = torch.from_numpy(audio_np).to(device=self.compute_device)

        acc_audio_data = torch.hstack([acc_audio_data, audio_torch])

        # Read the transcription.
        result = self.audio_model.transcribe(
          acc_audio_data,
          fp16=(self.compute_device == "cuda"),
          condition_on_previous_text=False,
          language=self.language,
        )
        text = result['text'].strip()

        # If we detected a pause between recordings, add a new item to our transcription.
        # Otherwise edit the existing one.
        if phrase_complete:
          transcription.append(text)
        else:
          transcription[-1] = text

        self.current_text = text
        self.hud_window.text = '\n'.join(transcription[-2:])

        # Clear the console to reprint the updated transcription.
        os.system('cls' if os.name=='nt' else 'clear')
        for seg in result['segments']:
          print(seg['start'], '->', seg['end'], seg['text'])
        # print(result)
        print(len(acc_audio_data))
        print(len(acc_audio_data)/self.source.SAMPLE_RATE/self.source.SAMPLE_WIDTH)

        for line in transcription:
          print(line)
        # Flush stdout.
        print('', end='', flush=True)

        # This is the last time we received new audio data from the queue.
        phrase_time = now
      except KeyboardInterrupt as e:
        print(e)
        break

    print("\n\nTranscription:")
    for line in transcription:
      print(line)

    return transcription

def main():
  args = parse_args()

  transcriber = Transcriber(args)
  transcriber.start_transribe_thread()
  transcriber.display_hud()
  transcriber.stop_transribe_thread()

if __name__ == "__main__":
  main()
