# Realtime onscreen transcription with Whisper

Transcribe your speech or the audio playing on your computer with Whisper.

## Installation

Install the following packages:

```shell
$ pip install -r requirement.txt
```

## Usage

1. (Optional) For transcribing the audio on your computer, setup a loopback device for your computer. Skip this step if you already setup a monitor device in other way. Note this script is only tested on Ubuntu 22.04.

    First list available devices for monitoring (this will also list your microphone).

    ```shell
    ./input-setup.sh
    ```

    Example output:

    ```
    Choose an audio output to monitor:
    2   alsa_input.microphone   module-alsa-card.c    s16le
    30   alsa_output.hdmi-stereo.monitor   module-alsa-card.c    s16le

    Usage:
      input-setup.sh OUTPUT_MONITOR
    ```

    Then create loopback device for your chosen device, for example for `alsa_output.hdmi-stereo.monitor`:

    ```shell
    $ ./input-setup.sh alsa_output.hdmi-stereo.monitor
    ```

    Then select the created loopback device "Transcribe-Output-Loopback" as your audio input from your system settings.

2. Launch the transcribe application.

    For transcribing the audio on your computer, use "pulse" as input. You can choose the input provider from speech-recognition and pyaudio. The difference is speech-recognition will surpress silence input.

    ```shell
    $ python transcribe.py --input pulse --input-provider speech-recognition --model base --no-faster-whisper
    ```

    For transcribing your speech, use your microphone as input. You can run the following command to choose from available devices.

    ```shell
    $ python transcribe.py --input-provider speech-recognition --model base --no-faster-whisper
    ```

    To run with faster whisper, omit the `--no-faster-whisper` option. Note for Cuda 12.x, you need to update your `LD_LIBRARY_PATH`, see [Troubleshoot - 1](#troubleshoot-1).

## Troubleshoot

1. To run with faster whisper with Cuda 12.x, udpate your `LD_LIBRARY_PATH` as follow. <a id="troubleshoot-1"></a>

    ```shell
    $ export pyvenv_path=YOUR_VENV_PATH # e.g. $HOME/.pyvenv/onscreen-transcription
    $ export pyvenv_py_version=YOUR_PYTHON_VERSION # python3.10
    $ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$pyvenv_path/lib64/$pyvenv_py_version/site-packages/nvidia/cublas/lib:$pyvenv_path/lib64/$pyvenv_py_version/site-packages/nvidia/cudnn/lib
    ```
