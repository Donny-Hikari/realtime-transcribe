#! python3.7

import sounddevice
import argparse
import os
import numpy as np
import speech_recognition as sr
import whisper
import torch

from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform

import tkinter as tk

# root = tk.Tk()
# root.attributes("-transparentcolor", "white")  # Set white color as transparent
# root.config(bg='white')  # Set background color to white (will be transparent)
# root.overrideredirect(True)  # Remove window decorations

# label = tk.Label(root, text="Hello, transparent world!", font=("Arial", 24))
# label.pack()

# root.mainloop()

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", default="medium", help="Model to use",
            choices=["tiny", "base", "small", "medium", "large", "tiny.en", "base.en", "small.en", "medium.en", "large-v3"])
  parser.add_argument("--input", default=None,
            help="Default input source.", type=str)
  parser.add_argument("--language", default=None,
            help="Default language.", type=str)

  parser.add_argument("--energy_threshold", default=1000,
            help="Energy level for mic to detect.", type=int)
  parser.add_argument("--record_timeout", default=1,
            help="How real time the recording is in seconds.", type=float)
  parser.add_argument("--phrase_timeout", default=3,
            help="How much empty space between recordings before we "
               "consider it a new line in the transcription.", type=float)
  args = parser.parse_args()
  return args

def main():
  args = parse_args()

  language = args.language
  sample_rate = 44100
  # The last time a recording was retrieved from the queue.
  phrase_time = None
  compute_device = "cuda" if torch.cuda.is_available() else "cpu"
  # Thread safe Queue for passing data from the threaded recording callback.
  data_queue = Queue()
  # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
  recorder = sr.Recognizer()
  recorder.energy_threshold = args.energy_threshold
  # Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
  recorder.dynamic_energy_threshold = False

  device_index = None
  if args.input is not None:
    for idx, name in enumerate(sr.Microphone.list_microphone_names()):
      if name == args.input:
        device_index = idx
        break

  if device_index is None:
    if args.input is not None:
      print(f'Invalid input device "{args.input}"')

    # Print the list of available audio devices
    print("Available input devices:")
    for idx, name in enumerate(sr.Microphone.list_microphone_names()):
      print(f"{idx}. {name}")

    if args.input is None:
      # Choose the index of the desired input device
      device_index = int(input("Enter the index of the input device you want to use: "))
    else:
      return

  source = sr.Microphone(sample_rate=sample_rate, device_index=device_index)

  # Load / Download model
  model = args.model
  audio_model = whisper.load_model(model, device=compute_device)

  record_timeout = args.record_timeout
  phrase_timeout = args.phrase_timeout

  transcription = ['']

  with source:
    recorder.adjust_for_ambient_noise(source)

  def record_callback(_, audio:sr.AudioData) -> None:
    """
    Threaded callback function to receive audio data when recordings finish.
    audio: An AudioData containing the recorded bytes.
    """
    # Grab the raw bytes and push it into the thread safe queue.
    data = audio.get_raw_data()
    data_queue.put(data)

  # Create a background thread that will pass us raw audio bytes.
  # We could do this manually but SpeechRecognizer provides a nice helper.
  recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

  # Cue the user that we're ready to go.
  print("Model loaded.\n")

  acc_audio_data = torch.zeros((0,), dtype=torch.float32, device=compute_device)
  while True:
    try:
      now = datetime.now()
      # Pull raw recorded audio from the queue.
      phrase_complete = False
      # If enough time has passed between recordings, consider the phrase complete.
      # Clear the current working audio buffer to start over with the new data.
      if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
        phrase_complete = True
        acc_audio_data = torch.zeros((0,), dtype=torch.float32, device=compute_device)

      data_queue.mutex.acquire()
      audio_data = b''.join(data_queue.queue)
      data_queue.queue.clear()
      data_queue.mutex.release()

      if len(audio_data) == 0:
        sleep(0.1)
        continue

      # Convert in-ram buffer to something the model can use directly without needing a temp file.
      # Convert data from 16 bit wide integers to floating point with a width of 32 bits.
      # Clamp the audio stream frequency to a PCM wavelength compatible default of 32768hz max.
      audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
      audio_torch = torch.from_numpy(audio_np).to(device=compute_device)

      acc_audio_data = torch.hstack([acc_audio_data, audio_torch])

      # Read the transcription.
      result = audio_model.transcribe(
        acc_audio_data,
        fp16=(compute_device == "cuda"),
        condition_on_previous_text=False,
        language=language,
      )
      text = result['text'].strip()

      # If we detected a pause between recordings, add a new item to our transcription.
      # Otherwise edit the existing one.
      if phrase_complete:
        transcription.append(text)
      else:
        transcription[-1] = text

      # Clear the console to reprint the updated transcription.
      os.system('cls' if os.name=='nt' else 'clear')
      for seg in result['segments']:
        print(seg['start'], '->', seg['end'], seg['text'])
      # print(result)
      print(len(acc_audio_data))
      print(len(acc_audio_data)/source.SAMPLE_RATE/source.SAMPLE_WIDTH)

      for line in transcription:
        print(line)
      # Flush stdout.
      print('', end='', flush=True)

      # This is the last time we received new audio data from the queue.
      phrase_time = now
    except KeyboardInterrupt:
      break

  print("\n\nTranscription:")
  for line in transcription:
    print(line)

if __name__ == "__main__":
  main()
