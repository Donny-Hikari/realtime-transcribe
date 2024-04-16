#! python3.7

import sounddevice
import argparse
import os
import numpy as np
import speech_recognition as sr
import whisper
import torch
import threading
import pyaudio
import signal

from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QLabel, QTextEdit
from PyQt5.QtGui import QFont, QFontMetrics, QCursor
from PyQt5.QtCore import QChildEvent, Qt, QTimer, QPoint, QRect, QSize

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", default="medium", help="Model to use",
            choices=["tiny", "base", "small", "medium", "large", "tiny.en", "base.en", "small.en", "medium.en", "large-v3"])
  parser.add_argument("--input", default=None,
            help="Default input source.", type=str)
  parser.add_argument("--input-provider", default="pyaudio",
            choices=["pyaudio", "speech-recognition"],
            help="Default input provider.", type=str)
  parser.add_argument("--language", default=None,
            help="Default language.", type=str)
  parser.add_argument("--font-size", default=30, type=int,
            help="Font size of HUD.")
  parser.add_argument("--no-fp16", action='store_true', default=False,
            help="Disable fp16.")

  parser.add_argument("--energy_threshold", default=500,
            help="Energy level for mic to detect.", type=int)
  parser.add_argument("--record_timeout", default=2,
            help="How real time the recording is in seconds.", type=float)
  parser.add_argument("--phrase_timeout", default=3,
            help="How much empty space between recordings before we "
               "consider it a new line in the transcription.", type=float)
  args = parser.parse_args()
  return args


class HUDText(QTextEdit):
  def __init__(self, font_size):
    super().__init__()

    self.setReadOnly(True)
    self.setAlignment(Qt.AlignLeft|Qt.AlignVCenter)
    self.setStyleSheet("background-color: rgba(0,0,0,0.5); color: white")
    self.setLineWrapMode(QTextEdit.WidgetWidth)
    font = QFont()
    font.setPointSize(font_size)
    self.setFont(font)
    self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    self.viewport().setCursor(Qt.CursorShape.ArrowCursor)

  def mousePressEvent(self, event):
    event.ignore()

  def mouseMoveEvent(self, event):
    event.ignore()

  def mouseReleaseEvent(self, event):
    event.ignore()

class HUD(QMainWindow):
  max_width_percentage = 0.4
  max_height_percentage = 0.16
  max_lines = 4
  padding = 10
  corner_spacing = 20

  def __init__(self, font_size):
    super().__init__()

    self.text = "Transcription started"

    # Set window attributes
    self.setWindowFlags(Qt.CustomizeWindowHint|Qt.FramelessWindowHint|Qt.WindowStaysOnTopHint|Qt.WindowDoesNotAcceptFocus)
    self.setAttribute(Qt.WA_TranslucentBackground)

    # Set window opacity
    self.setWindowOpacity(0.5)

    # Set central widget
    central_widget = QWidget()
    layout = QHBoxLayout(central_widget)

    self.text_widget = HUDText(font_size)
    self.text_widget.setParent(central_widget)
    layout.addWidget(self.text_widget)

    self.setStyleSheet(f"padding: {self.padding}px;")
    self.setCentralWidget(central_widget)

    # Limit window size to 60% of screen width
    screen_geometry = QApplication.desktop().screenGeometry()
    max_width = int(self.max_width_percentage * screen_geometry.width())
    max_height = int(self.max_height_percentage * screen_geometry.height())
    self.setFixedWidth(max_width)
    self.setFixedHeight(max_height) # sadly qt is unable to measure text height properly, so we have to set height ratio to screen and leave it ugly
    self.move(screen_geometry.width()-max_width-self.corner_spacing, screen_geometry.height()-max_height-self.corner_spacing)

    # Start updating the text periodically
    self.update_text_timer = QTimer(self)
    self.update_text_timer.timeout.connect(self.updateTextWidget)
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

  def updateTextWidget(self):
    self.text_widget.setText(self.text)

    vertical_scrollbar = self.text_widget.verticalScrollBar()
    vertical_scrollbar.setValue(vertical_scrollbar.maximum())

  def update_text(self, text):
    self.text = text


class AudioInputProvider:
  def list_input_devices(self):
    raise NotImplementedError

  def init_input_device(self, device_index):
    raise NotImplementedError

  def start_record(self):
    raise NotImplementedError

  def stop_record(self):
    raise NotImplementedError

  def phrase_cut_off(self, acc_data, new_data):
    raise NotImplementedError

class SpeechRecognitionAudioProvider(AudioInputProvider):
  def __init__(self, args, data_queue, sample_rate):
    self.sample_rate = sample_rate
    self.energy_threshold = args.energy_threshold
    self.record_timeout = args.record_timeout
    self.phrase_timeout = args.phrase_timeout

    self.data_queue = data_queue
    self.stop_listening = None

  def list_input_devices(self):
    return sr.Microphone.list_microphone_names()

  def init_input_device(self, device_index):
    # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
    self.recorder = sr.Recognizer()
    self.recorder.energy_threshold = self.energy_threshold
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
    self.recorder.dynamic_energy_threshold = False

    self.source = sr.Microphone(sample_rate=self.sample_rate, device_index=device_index)

    with self.source:
      self.recorder.adjust_for_ambient_noise(self.source)

  def start_record(self):
    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    self.stop_listening = self.recorder.listen_in_background(self.source, self.record_callback, phrase_time_limit=self.record_timeout)

    self.phrase_time = None

  def stop_record(self):
    if self.stop_listening is None:
      raise RuntimeError("Recording not started")

    self.stop_listening(True)

    del self.phrase_time

  def phrase_cut_off(self, acc_data, new_data):
    now = datetime.now()

    # If enough time has passed between recordings, consider the phrase complete.
    # Clear the current working audio buffer to start over with the new data.
    if self.phrase_time and now - self.phrase_time > timedelta(seconds=self.phrase_timeout):
      # This is the last time we received new audio data from the queue.
      self.phrase_time = now # should set after the transcription is done
      return len(acc_data)
    else:
      return 0

  def record_callback(self, _, audio: sr.AudioData) -> None:
    """
    Threaded callback function to receive audio data when recordings finish.
    audio: An AudioData containing the recorded bytes.
    """
    # Grab the raw bytes and push it into the thread safe queue.
    data = audio.get_raw_data()
    self.data_queue.put(data)

class PyAudioProvider(AudioInputProvider):
  def __init__(self, args, data_queue, sample_rate):
    self.audio = pyaudio.PyAudio()

    self.audio_format = pyaudio.paInt16  # Format of the audio samples
    self.audio_channels = 1  # Number of audio channels (1 for mono, 2 for stereo)
    self.sample_rate = sample_rate  # Sample rate (samples per second)
    self.sample_size = self.audio.get_sample_size(self.audio_format)
    self.audio_chunk = 10240  # Number of frames per buffer

    self.device_index = None
    self.stop_event = threading.Event()

    self.data_queue = data_queue

  def list_input_devices(self):
    for idx in range(self.audio.get_device_count()):
      device_info = self.audio.get_device_info_by_index(idx)
      yield device_info['name']

  def init_input_device(self, device_index):
    self.device_index = device_index

  def start_record(self):
    self.stop_event.clear()

    self.record_thread = threading.Thread(target=self._record_audio)
    self.record_thread.start()

  def stop_record(self):
    self.stop_event.set()
    self.record_thread.join()

  def phrase_cut_off(self, acc_data, new_data):
    if len(acc_data) + len(new_data) > self.sample_rate*self.sample_size*60:
      return len(acc_data) // 2
    else:
      return 0

  def _record_audio(self):
    def stream_callback(in_data, frame_count, time_info, status):
        self.data_queue.put(in_data)
        return (None, pyaudio.paContinue)

    # Open audio stream with the selected input device
    stream = self.audio.open(
      format=self.audio_format,
      channels=self.audio_channels,
      rate=self.sample_rate,
      input=True,
      input_device_index=self.device_index,
      frames_per_buffer=self.audio_chunk,
      stream_callback=stream_callback,
    )

    while not self.stop_event.is_set() and stream.is_active():
      sleep(0.1)

    # Close the audio stream
    stream.stop_stream()
    stream.close()

class Transcriber():
  def __init__(self, args):
    self.args = args
    self.language = args.language
    self.sample_rate = whisper.audio.SAMPLE_RATE
    # The last time a recording was retrieved from the queue.
    self.compute_device = "cuda" if torch.cuda.is_available() else "cpu"
    self.model_name = args.model

    # Thread safe Queue for passing data from the threaded recording callback.
    self.data_queue = Queue()
    self.transribe_thread = None
    self.stop_event = threading.Event()
    self.hud_window = None

    if args.input_provider == "speech-recognition":
      self.input_provider = SpeechRecognitionAudioProvider(args=self.args, data_queue=self.data_queue, sample_rate=self.sample_rate)
    elif args.input_provider == "pyaudio":
      self.input_provider = PyAudioProvider(args=self.args, data_queue=self.data_queue, sample_rate=self.sample_rate)

    self.init_input_device(args)

    # Load / Download model
    self.audio_model = whisper.load_model(self.model_name, device=self.compute_device)

    # Cue the user that we're ready to go.
    print("Model loaded.\n")

  def init_input_device(self, args):
    device_index = None

    if args.input is not None:
      for idx, name in enumerate(self.input_provider.list_input_devices()):
        if name == args.input:
          device_index = idx
          break

    if device_index is None:
      # Print the list of available audio devices
      print("Available input devices:")
      for idx, name in enumerate(self.input_provider.list_input_devices()):
        print(f"{idx}. {name}")

      if args.input is None:
        # Choose the index of the desired input device
        device_index = int(input("Enter the index of the input device you want to use: "))
      else:
        raise RuntimeError(f'Invalid input device "{args.input}"')

    self.input_provider.init_input_device(device_index)

  def display_hud(self):
    app = QApplication([])

    def handle_signal(sig, frame):
      app.quit()
    signal.signal(signal.SIGINT, handle_signal)

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

    self.input_provider.start_record()

    acc_audio_data = torch.zeros((0,), dtype=torch.float32, device=self.compute_device)
    while not self.stop_event.is_set():
      try:
        self.data_queue.mutex.acquire()
        audio_data = b''.join(self.data_queue.queue)
        self.data_queue.queue.clear()
        self.data_queue.mutex.release()

        if len(audio_data) == 0:
          sleep(0.1)
          continue

        phrase_cut_off = self.input_provider.phrase_cut_off(acc_audio_data, audio_data)
        acc_audio_data = acc_audio_data[phrase_cut_off:]

        # Convert in-ram buffer to something the model can use directly without needing a temp file.
        # Convert data from 16 bit wide integers to floating point with a width of 32 bits.
        # Clamp the audio stream frequency to a PCM wavelength compatible default of 32768hz max.
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        audio_torch = torch.from_numpy(audio_np).to(device=self.compute_device)

        acc_audio_data = torch.hstack([acc_audio_data, audio_torch])

        # Read the transcription.
        result = self.audio_model.transcribe(
          acc_audio_data,
          fp16=not self.args.no_fp16 and (self.compute_device == "cuda"),
          condition_on_previous_text=False,
          language=self.language,
        )
        wrapped_text = '\n'.join(map(lambda x: x['text'], result['segments']))

        # If we detected a pause between recordings, add a new item to our transcription.
        # Otherwise edit the existing one.
        if phrase_cut_off > 0:
          transcription.append(wrapped_text)
        else:
          transcription[-1] = wrapped_text

        if self.hud_window is not None:
          self.hud_window.update_text(wrapped_text)

        # Clear the console to reprint the updated transcription.
        os.system('cls' if os.name=='nt' else 'clear')
        for seg in result['segments']:
          print('%.2f' % seg['start'], '->', '%.2f' % seg['end'], seg['text'])

        for line in transcription:
          print(line)
        # Flush stdout.
        print('', end='', flush=True)
      except KeyboardInterrupt as e:
        break

    print("\n\nTranscription:")
    for line in transcription:
      print(line)

    self.input_provider.stop_record()

    return transcription

def main():
  args = parse_args()

  transcriber = Transcriber(args)
  transcriber.start_transribe_thread()
  transcriber.display_hud()
  transcriber.stop_transribe_thread()

if __name__ == "__main__":
  main()
