# Realtime onscreen transcription with Whisper

Transcribe your speech or the audio playing on your computer with Whisper in realtime, and show the captions on your screen.

https://github.com/Donny-Hikari/realtime-transcribe/assets/22200374/082a7b41-ace2-428b-a886-9c526d95aa44

## Installation

Install the following packages:

```shell
$ pip install -r requirement.txt
```

## Usage

### Transcribe your speech

Run the following command to start transcribing your speech:

```shell
$ python transcribe.py --input-provider speech-recognition --model base --no-faster-whisper
```

A list of audio input device will be displayed. Choose your microphone to start transcribing.

Some options:

1. You can choose the `--input-provider` from "speech-recognition" and "pyaudio". The difference is speech-recognition will surpress silence input.

2. To run with faster whisper, omit the `--no-faster-whisper` option. Note for Cuda 12.x, you need to update your `LD_LIBRARY_PATH`, see [Troubleshoot - 1](#troubleshoot-1).

3. For better percision, use the `--language` option to specify the input language (in [ISO 639-1 codes](https://en.wikipedia.org/wiki/List_of_ISO_639_language_codes)).

### Transcribe the audio output on your computer

1. (Optional) Setup monitor device for audio output.

    Setup a loopback device for your audio output. Skip this step if you already setup a monitor device in other way.

    First list available devices for monitoring (this will also list your microphone).

    ```shell
    $ pactl list sources short
    ```

    Example output:

    ```
    2   alsa_input.microphone   module-alsa-card.c    s16le 1ch 44100Hz   SUSPENDED
    30   alsa_output.hdmi-stereo.monitor   module-alsa-card.c    s16le 2ch 44100Hz RUNNING
    ```

    Then set the pulse source to your chosen device, for example "alsa_output.hdmi-stereo.monitor".
    ```shell
    $ export PULSE_SOURCE=alsa_output.hdmi-stereo.monitor
    ```

2. Start transcribing.

    For transcribing from the device chosen in step 1, use "pulse" as input.

    ```shell
    $ python transcribe.py --input pulse --input-provider speech-recognition --model base --no-faster-whisper
    ```

## Troubleshoot

1. To run with faster whisper with Cuda 12.x, udpate your `LD_LIBRARY_PATH` as follow. <a id="troubleshoot-1"></a>

    ```shell
    $ export pyvenv_path=YOUR_VENV_PATH # e.g. $HOME/.pyvenv/onscreen-transcription
    $ export pyvenv_py_version=YOUR_PYTHON_VERSION # python3.10
    $ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$pyvenv_path/lib64/$pyvenv_py_version/site-packages/nvidia/cublas/lib:$pyvenv_path/lib64/$pyvenv_py_version/site-packages/nvidia/cudnn/lib
    ```
