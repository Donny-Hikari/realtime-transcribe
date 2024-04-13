import pyaudio
import whisper
import torch
import numpy as np
import sounddevice
import threading
from queue import Queue
import argparse
import time
from faster_whisper import WhisperModel

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", default="medium", help="Model to use",
            choices=["tiny", "base", "small", "medium", "large", "tiny.en", "base.en", "small.en", "medium.en", "large-v3"])
  parser.add_argument("--input", default=None,
            help="Default input source.", type=str)
  parser.add_argument("--language", default=None,
            help="Default language.", type=str)
  args = parser.parse_args()
  return args

def main():
  args = parse_args()

  # Initialize PyAudio
  audio = pyaudio.PyAudio()

  device_index = None
  if args.input is not None:
    for idx in range(audio.get_device_count()):
      device_info = audio.get_device_info_by_index(idx)
      if device_info['name'] == args.input:
        device_index = idx
        break

  if device_index is None:
    if args.input is not None:
      print(f'Invalid input "{args.input}"')

    # Print the list of available audio devices
    for idx in range(audio.get_device_count()):
      device_info = audio.get_device_info_by_index(idx)
      print(f"Device {idx}: {device_info['name']}")

    if args.input is None:
      # Choose the index of the desired input device
      device_index = int(input("Enter the index of the input device you want to use: "))
    else:
      return

  # Parameters
  language = args.language
  audio_format = pyaudio.paInt16  # Format of the audio samples
  audio_channels = 1  # Number of audio channels (1 for mono, 2 for stereo)
  sample_rate = 44100  # Sample rate (samples per second)
  sample_size = audio.get_sample_size(audio_format)
  audio_chunk = 1024  # Number of frames per buffer
  model = args.model
  compute_device = "cpu"
  compute_type = "float16" if compute_device == "cuda" else "float32"

  audio_model = WhisperModel(model, device=compute_device, compute_type=compute_type)
  # audio_model = whisper.load_model(model, device=compute_device)

  audio_queue = Queue()
  stop_event = threading.Event()

  def record_audio():
    # Open audio stream with the selected input device
    stream = audio.open(
      format=audio_format,
      channels=audio_channels,
      rate=sample_rate,
      input=True,
      input_device_index=device_index,
      frames_per_buffer=audio_chunk,
    )

    while not stop_event.is_set():
      data = stream.read(audio_chunk, exception_on_overflow=False)
      audio_queue.put(data)

    # Close the audio stream
    stream.stop_stream()
    stream.close()

  record_thread = threading.Thread(target=record_audio)
  record_thread.start()
  print("Recording...")

  # Record audio data
  # acc_data = torch.zeros((0,), dtype=torch.float32, device=compute_device)
  acc_data = np.zeros((0,), dtype=np.float32)
  while True:
    try:
      audio_queue.mutex.acquire()
      audio_data = b''.join(audio_queue.queue)
      audio_queue.queue.clear()
      audio_queue.mutex.release()

      if len(audio_data) == 0:
        time.sleep(0.1)
        continue

      audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
      # audio_torch = torch.from_numpy(audio_np).to(device=compute_device)

      # acc_data = torch.hstack([acc_data, audio_torch])
      acc_data = np.hstack([acc_data, audio_np])

      # Read the transcription.
      segments, info = audio_model.transcribe(
        acc_data,
        vad_filter=True,
        language=language,
      )
      segments = list(segments)
      text = ''.join(map(lambda x: x.text, segments)).strip()

      for seg in segments:
        print(seg.start, seg.end, seg.text)
      print(text)
      print(len(acc_data)/sample_rate/sample_size)
    except KeyboardInterrupt:
      break

  stop_event.set()
  record_thread.join()

  audio.terminate()

if __name__ == "__main__":
  main()
