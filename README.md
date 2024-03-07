# Scene Scribe

## Getting Started

* Get an access token at https://huggingface.co/settings/tokens.

* install `ffmpeg`

```bash
pip install -r requirements.txt
```

```bash

HF_TOKEN=python -m transcribe path/to/video.mp4
```

## Todo

does condense_segments need to happen where it is? It rejoins segments split by scene.

## Basic Idea

Use yt-dlp to retrieve video from CABQ website

Use ffmpeg to extract audio and convert to `.wav`.

```bash
ffmpeg -i input_video.mp4 -vn -acodec pcm_s16le -ar 44100 -ac 2 output_audio.wav
```

Use `whisper` for transcript

Use `pyannote.audio` to diariaze

Use ffmpeg to extract a screenshot from the video for each speaker turn

```bash
ffmpeg -ss HH:MM:SS.ms -i input_video.mp4 -frames:v 1 output_screenshot.jpg
```
