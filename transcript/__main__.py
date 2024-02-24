from pathlib import Path
import hashlib
import subprocess
from tempfile import NamedTemporaryFile
import os
from typing import List, Tuple
import math
import datetime


import whisper
import click
from ncls import NCLS
import imagehash
from PIL import Image
import scipy


from sqlalchemy.orm import Mapped, mapped_column, relationship, Session
from sqlalchemy import Text, Integer, ForeignKey, Float
from sqlalchemy import create_engine
from sqlalchemy import select

from transcript.video import Base, Video
from transcript.frames import Frame
import transcript as tr
from transcript.logging import log


HF_ACCESS_TOKEN = os.environ["HF_TOKEN"]
SCENE_CHANGE_DHASH_DELTA = 14


class Speaker(Base):
    """A Named Speaker"""

    __tablename__ = "speaker"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(Text)

    turns: Mapped["Turn"] = relationship(back_populates="speaker")  # turn.speaker


class Turn(Base):
    """A speaker's turn"""

    __tablename__ = "turn"
    id: Mapped[int] = mapped_column(primary_key=True)
    speaker_id: Mapped[int] = mapped_column(ForeignKey("speaker.id"))
    start_ms: Mapped[int] = mapped_column(Integer)
    stop_ms: Mapped[int] = mapped_column(Integer)

    speaker: Mapped["Speaker"] = relationship(back_populates="turns")  # speaker.turns


class SpeakerChunk(Base):
    """A continuous segment assigned to a single speaker"""

    __tablename__ = "speaker_chunk"
    id: Mapped[int] = mapped_column(primary_key=True)
    speaker_id: Mapped[int] = mapped_column(ForeignKey("speaker.id"))
    start_ms: Mapped[int] = mapped_column(Integer)
    stop_ms: Mapped[int] = mapped_column(Integer)
    transcribed: Mapped[bool] = mapped_column(Integer, default=False)


class Segment(Base):
    """A transcribed segment of text"""

    __tablename__ = "segment"
    id: Mapped[int] = mapped_column(primary_key=True)
    start_ms: Mapped[int] = mapped_column(Integer)
    stop_ms: Mapped[int] = mapped_column(Integer)
    speech_prob: Mapped[float] = mapped_column(Float)
    speaker_id: Mapped[int] = mapped_column(ForeignKey("speaker.id"), nullable=True)
    text: Mapped[str] = mapped_column(Text)


class Screencap(Base):
    __tablename__ = "screencap"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(Text)
    timestamp_ms: Mapped[int] = mapped_column(Integer)


class Scene(Base):
    __tablename__ = "scene"
    id: Mapped[int] = mapped_column(primary_key=True)
    start_ms: Mapped[int] = mapped_column(Integer)
    stop_ms: Mapped[int] = mapped_column(Integer)


def extract_audio_as_mp3(
    video_path: Path, root_dir: Path, start=None, stop=None
) -> Path:
    """returns path to audio file"""

    f = NamedTemporaryFile(delete=False, dir=root_dir, suffix=f".mp3")
    try:
        output_path = Path(f.name).resolve()
        cmd = [
            "ffmpeg",
            "-y",  # overwrite existing file
            "-i",
            str(video_path),
            "-vn",
            "-ar",
            "44100",
            "-ac",
            "2",
            "-b:a",
            "192k",
        ]
        if start is not None:
            cmd += [
                "-ss",
                str(float(start)),
            ]
        if stop is not None:
            cmd += [
                "-to",
                str(float(stop)),
            ]
        cmd += [str(output_path)]
        log(f"running {cmd}")
        subprocess.run(cmd, capture_output=True).check_returncode()
    except Exception as e:
        os.unlink(f)
        raise e
    return output_path


def extract_audio_as_wav(video_path: Path, root_dir: Path) -> Path:
    """returns path to audio file"""

    f = NamedTemporaryFile(delete=False, dir=root_dir, suffix=".wav")
    try:
        output_path = Path(f.name).resolve()
        cmd = [
            "ffmpeg",
            "-y",  # overwrite existing file
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "44100",
            "-ac",
            "2",
            str(output_path),
        ]
        log(f"running {cmd}")
        subprocess.run(cmd, capture_output=True).check_returncode()
    except Exception as e:
        os.unlink(f)
        raise e
    return output_path


def best_torch_device() -> str:
    import torch

    """returns a string appropriate to pass to torch.device()"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def diarize(session: Session, video_path: Path, root_dir: Path):
    import torch
    import torchaudio
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.utils.hook import ProgressHook

    video = session.scalars(select(Video)).one()

    wav_path = extract_audio_as_wav(video_path, root_dir)
    log(f"diarize: got audio at {wav_path}")

    try:
        log(f"diarize: create pipeline...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HF_ACCESS_TOKEN,
        )

        device_str = best_torch_device()
        log(f'diarize: use torch device "{device_str}"')
        log(f"diarize: move pipeline to device")
        pipeline.to(torch.device(device_str))

        # apply pretrained pipeline

        with ProgressHook() as hook:
            # this torchaudio.load seems necessary for reasonable performance in the pipeline
            log(f"diarize: torchaudio.load")
            waveform, sample_rate = torchaudio.load(str(wav_path))
            log(f"diarize: apply pipeline")
            diarization = pipeline(
                {"waveform": waveform, "sample_rate": sample_rate}, hook=hook
            )
    except Exception as e:
        raise e
    finally:
        log(f"diarize: delete {wav_path}")
        wav_path.unlink(missing_ok=True)

    log(f"diarize: add speakers")
    for _, _, speaker_name in diarization.itertracks(yield_label=True):
        speaker = session.scalars(
            select(Speaker).where(Speaker.name == speaker_name)
        ).one_or_none()
        if speaker is None:
            speaker = Speaker(name=speaker_name)
            session.add(speaker)
            session.commit()
            log(f"new speaker video={video.id} name={speaker_name} id={speaker.id}")

    log(f"diarize: update state")
    for turn, _, speaker_name in diarization.itertracks(yield_label=True):
        # print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker={speaker_name}")
        # start=0.2s stop=1.5s speaker_0
        # start=1.8s stop=3.9s speaker_1
        # start=4.2s stop=5.7s speaker_0
        # ...

        speaker = session.scalars(
            select(Speaker).where(Speaker.name == speaker_name)
        ).one()

        session.add(
            Turn(
                start_ms=int(turn.start * 1000 + 0.5),  # convert times to ms
                stop_ms=int(turn.end * 1000 + 0.5),
                speaker_id=speaker.id,
            )
        )

    video.diarized = True

    log(f"diarize: commit state")
    session.commit()


def remove_diarize(session: Session):
    video = session.scalars(select(Video)).one()
    turns = session.scalars(select(Turn)).all()
    chunks = session.scalars(select(SpeakerChunk)).all()
    video.diarized = False
    for turn in turns:
        session.delete(turn)
    for chunk in chunks:
        session.delete(chunk)
    session.commit()


def remove_scenes(session: Session):
    video = session.scalars(select(Video)).one()
    scenes = session.scalars(select(Scene)).all()
    for scene in scenes:
        session.delete(scene)
    video.scenes_detected = False
    session.commit()


def detect_scenes(session: Session, video_path, root_dir: Path):

    video = session.scalars(select(Video)).one()

    hashes = [
        (imagehash.hex_to_hash(f.dhash), f.actual_ms / 1000)
        for f in session.scalars(select(Frame).order_by(Frame.target_ms)).all()
    ]

    x = [b[0] - a[0] for a, b in zip(hashes[:-1], hashes[1:])]
    peaks, props = scipy.signal.find_peaks(x, prominence=SCENE_CHANGE_DHASH_DELTA)
    peaks = peaks * tr.frames.FRAME_INTERVAL

    # peak[0] is ts[1] - ts[0], so shift all these by 1 frame interval
    # to get when the scene actually changed
    # print(peaks + FRAME_INTERVAL)
    # print(props["prominences"])
    # print(props["left_bases"] * FRAME_INTERVAL + FRAME_INTERVAL)
    # print(props["right_bases"] * FRAME_INTERVAL + FRAME_INTERVAL)
    peaks = peaks + tr.frames.FRAME_INTERVAL

    start = 0
    for ts in peaks:
        ts_ms = int(ts * 1000 + 0.5)
        session.add(Scene(start_ms=start, stop_ms=ts_ms))
        start = ts_ms
    video.scenes_detected = True
    session.commit()


def make_speaker_chunks(session: Session):

    MERGE_GAP_MS = 10000

    earliest_start_turn = session.scalars(
        select(Turn).order_by(Turn.start_ms).limit(1)
    ).one()

    last_finish_turn = session.scalars(
        select(Turn).order_by(Turn.stop_ms.desc()).limit(1)
    ).one()

    log(f"make_speaker_chunks: detect turn edges...")
    cursor = earliest_start_turn.start_ms
    edges = [(cursor, earliest_start_turn, "start")]
    while cursor < last_finish_turn.stop_ms:

        # next change is either a turn stopping or starting
        next_start_turn = session.scalars(
            select(Turn).where(Turn.start_ms > cursor).order_by(Turn.start_ms).limit(1)
        ).one_or_none()

        next_stop_turn = session.scalars(
            select(Turn).where(Turn.stop_ms > cursor).order_by(Turn.stop_ms).limit(1)
        ).one_or_none()

        if next_start_turn is not None and next_stop_turn is not None:
            if next_start_turn.start_ms < next_stop_turn.stop_ms:
                # update cursor
                cursor = next_start_turn.start_ms
                # the next thing is that a turn starts
                edges += [(cursor, next_start_turn, "start")]

            elif next_stop_turn.stop_ms < next_start_turn.start_ms:
                # update cursor
                cursor = next_stop_turn.stop_ms
                # next thing is that a turn stops
                edges += [(cursor, next_stop_turn, "stop")]

            else:  # next stop and next start are at the same time
                # update cursor
                cursor = next_stop_turn.stop_ms
                edges += [(cursor, next_stop_turn, "stop")]
                edges += [(cursor, next_start_turn, "start")]
        elif next_start_turn is not None:
            # the next thing is that a turn starts
            cursor = next_start_turn.start_ms
            edges += [(cursor, next_start_turn, "start")]
        elif next_stop_turn is not None:
            # next thing is that a turn stops
            cursor = next_stop_turn.stop_ms
            edges += [(cursor, next_stop_turn, "stop")]
        else:
            assert False

    # print(edges)
    # print()

    log(f"make_speaker_chunks: convert {len(edges)} edges into regions...")
    regions = []
    active = set()
    region_start = None
    for ts, turn, which in edges:
        if active:
            regions += [(region_start, ts, set(active))]
        region_start = ts
        if which == "start":
            active.add(turn)
        elif which == "stop":
            active.remove(turn)

    # print([(start, stop, {t.id for t in active}) for start, stop, active in regions])
    # print()

    region_turns = []
    # assign each region to whatever turn out of the active turns is longest
    log(f"make_speaker_chunks: assign a dominant turn to {len(regions)} regions...")
    for region in regions:
        start, stop, turns = region

        ## break ties by lower turn id
        longest_turn = max(turns, key=lambda t: (t.stop_ms - t.start_ms, t.id))
        region_turns += [(start, stop, longest_turn)]

    # print([(start, stop, turn.id) for start, stop, turn in region_turns])
    # print()

    # merge nearby regions with the same speaker
    log(f"make_speaker_chunks: merge {len(region_turns)} regions by speaker...")
    changed = True
    chunks = [(start, stop, t.speaker_id) for start, stop, t in region_turns]
    while changed:
        changed = False
        new_chunks = []
        ri = 0
        while ri < len(chunks):
            rj = ri + 1
            if rj == len(chunks):
                new_chunks += [chunks[ri]]
            else:
                start_i, stop_i, s_i = chunks[ri]
                start_j, stop_j, s_j = chunks[rj]
                if start_j - stop_i < MERGE_GAP_MS and s_i == s_j:
                    new_chunks += [(start_i, stop_j, s_i)]
                    changed = True
                    ri += 1
                else:
                    new_chunks += [chunks[ri]]
            ri += 1
        chunks = new_chunks

    log(f"make_speaker_chunks: merged by speaker into {len(chunks)} chunks")

    # remove any extremely short chunks
    # these chunks can cause whisper to fail to detect
    # the audio codec
    chunks = [c for c in chunks if c[1] - c[0] > 52]  # 52 ms

    avg_chunk_length = sum(c[1] - c[0] for c in chunks) / len(chunks)
    whisper_length = sum(max(c[1] - c[0], 30) for c in chunks)
    log(
        f"make_speaker_chunks: removed short chunks into {len(chunks)} chunks (avg. {avg_chunk_length:.1f}ms wl={whisper_length:.1f}ms)"
    )
    # print(chunks)
    # sys.exit(1)
    # print()

    # test some stuff about chunks
    for i, chunk in enumerate(chunks):
        if i > 0 and i < len(chunks) - 1:
            if not (chunk[0] >= chunks[i - 1][1]):
                print(chunks[i - 1], chunk)
                assert False
            assert chunk[1] <= chunks[i + 1][0]

    for chunk in chunks:
        sc = SpeakerChunk(
            start_ms=chunk[0],
            stop_ms=chunk[1],
            speaker_id=chunk[2],
        )

        existing = session.scalars(
            select(SpeakerChunk)
            .where(SpeakerChunk.start_ms == sc.start_ms)
            .where(SpeakerChunk.stop_ms == sc.stop_ms)
            .where(SpeakerChunk.speaker_id == sc.speaker_id)
        ).one_or_none()

        if existing is None:
            session.add(sc)
    session.commit()


def detect_audio_codec(video_path: Path) -> str:

    cp = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
    )
    if cp.returncode != 0:
        raise RuntimeError(cp.stderr)
    else:
        return cp.stdout.decode("utf-8").strip()


def extract_raw_audio(video_path: Path, root_dir: Path, start=None, stop=None) -> Path:
    """returns path to audio file"""

    codec = detect_audio_codec(video_path)

    f = NamedTemporaryFile(delete=False, dir=root_dir, suffix=f".{codec}")
    try:
        output_path = Path(f.name).resolve()
        cmd = [
            "ffmpeg",
            "-y",  # overwrite existing file
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "copy",
        ]
        if start is not None:
            cmd += [
                "-ss",
                str(float(start)),
            ]
        if stop is not None:
            cmd += [
                "-to",
                str(float(stop)),
            ]
        cmd += [str(output_path)]
        log(f"running {cmd}")
        subprocess.run(cmd, capture_output=True).check_returncode()
    except Exception as e:
        os.unlink(f)
        raise e
    return output_path


def transcribe(session: Session, video_path: Path, root_dir: Path):
    """Extract each turn from the video and transcribe it

    Whisper seems to have problems for long transcriptions, so we try to produce
    approximately 10-minute chunks, using gaps between turns to do it

    If two speakers are in the same chunk, Whisper may return a segment
    that contains text from both speakers, and it's hard to split that
    back out.

    First, combine all consecutive closely-spaced turns with the same speaker
    into larger chunks
    """

    video = session.scalars(select(Video)).one()
    chunks = session.scalars(
        select(SpeakerChunk)
        .where(SpeakerChunk.transcribed == False)
        .order_by(SpeakerChunk.start_ms)
    ).all()

    log("transcribe: load model")
    model = whisper.load_model(
        "small.en",
        # device="mps"
    )

    for ci, chunk in enumerate(chunks):
        log(f"transcribe: chunk {ci+1}/{len(chunks)}: extract audio segment")

        try:
            audio_path = extract_raw_audio(
                video_path,
                root_dir,
                start=chunk.start_ms / 1000,
                stop=chunk.stop_ms / 1000,
            )

            log(f"transcribe: chunk {ci+1}/{len(chunks)}: transcribe")
            results = model.transcribe(
                str(audio_path),
                initial_prompt="Albuquerque, Sandia, Kirtland Air Force Base, Councilor, executive session, Bernalillo",
                verbose=False,  # simple progress bar
            )
        except Exception as e:
            raise e
        finally:
            log(f"transcribe: chunk {ci+1}/{len(chunks)}: remove {audio_path}")
            audio_path.unlink(missing_ok=True)

        # exclude segments that are probably not speech
        # a higher number here is less strict (i.e., more likely to include
        # segments that are not speech)

        segments = []
        for s in results["segments"]:
            if s["no_speech_prob"] < 0.90:
                segments += [s]
            else:
                log(f'skipped "{s["text"]}": no_speech_prob={s["no_speech_prob"]}')

        log(f"transcribe: chunk {ci+1}/{len(chunks)}: add {len(segments)} segments")
        for seg in segments:

            # if the segment is very short, whisper pads it out to 30s,
            # so its possible for the segment length to be longer than the chunk
            # length. Cap the end of the segment at the end of the chunk
            start_ms = int(seg["start"] * 1000 + 0.5) + chunk.start_ms
            stop_ms = min(int(seg["end"] * 1000 + 0.5) + chunk.start_ms, chunk.stop_ms)
            session.add(
                Segment(
                    start_ms=start_ms,
                    stop_ms=stop_ms,
                    text=seg["text"].strip(),
                    speech_prob=(1.0 - seg["no_speech_prob"]),
                )
            )
        chunk.transcribed = True
        session.commit()
    log(f"transcribe: mark video transcription commplete")
    video.transcribed = True
    session.commit()


def remove_transcribe(session: Session):
    video = session.scalars(select(Video)).one()
    segments = session.scalars(select(Segment)).all()
    chunks = session.scalars(
        select(SpeakerChunk).where(SpeakerChunk.transcribed == True)
    ).all()
    video.transcribed = False
    for segment in segments:
        session.delete(segment)
    for chunk in chunks:
        chunk.transcribed = False
    session.commit()


def join_segments(s1: str, s2: str) -> str:
    s1 = s1.strip()
    s2 = s2.strip()
    if s1[-1] != " " and s2[0] != " ":
        return s1 + " " + s2
    elif s1[-1] == " " and s2[0] == " ":
        return s1 + s2[1:]
    else:
        return s1 + s2


def condense_segments(session: Session) -> List[Tuple]:
    segments = session.scalars(
        select(Segment).where(Segment.speaker_id != None).order_by(Segment.stop_ms)
    ).all()

    # merge adjacent segments with the same speaker
    segments = [
        (s.start_ms, s.stop_ms, s.speaker_id, s.text, s.speech_prob) for s in segments
    ]

    for i, seg in enumerate(segments):
        if i > 0 and i < len(segments) - 1:
            assert segments[i - 1][1] <= seg[0]
            if not (seg[1] <= segments[i + 1][0]):
                print(seg[0:2], segments[i + 1][0:2])
                assert False

    changed = True
    while changed:
        changed = False
        new_segments = []
        si = 0
        while si < len(segments):
            s1 = segments[si]
            if si == len(segments) - 1:
                new_segments += [s1]
                si += 1
            else:
                s2 = segments[si + 1]
                if s1[2] == s2[2]:  # same speaker
                    new_segments += [
                        (
                            s1[0],
                            s2[1],
                            s1[2],
                            join_segments(s1[3], s2[3]),
                            min(s1[4], s2[4]),
                        )
                    ]
                    si += 2
                    changed = True
                else:
                    new_segments += [s1]
                    si += 1
        segments = new_segments

    for i, seg in enumerate(segments):
        if i > 0 and i < len(segments) - 1:
            assert segments[i - 1][1] <= seg[0]
            if not (seg[1] <= segments[i + 1][0]):
                print(seg[0:2], segments[i + 1][0:2])
                assert False
    return segments


def hh_mm_ss_ms(t: float):
    ss, ms = divmod(t, 1)
    mm, ss = divmod(int(ss), 60)
    hh, mm = divmod(mm, 60)
    ms = f"{ms:.1f}"[2:]
    return f"{hh}:{mm:02d}:{ss:02d}.{ms}"


def text_output(session: Session, segments):
    c = f"Transcript"
    c += "\n=========="

    for start, stop, speaker_id, text, speech_prob in segments:

        c += f"\n\n{hh_mm_ss_ms(start/1000)} - {hh_mm_ss_ms(stop/1000)}"  # convert to s
        c += "\n----------"

        c += "\n" + text

    c += "\n"
    return c


def html_output(session: Session, segments, root_dir: Path, title=None) -> str:

    c = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Document</title>
<style>
    body {
        font-family: sans-serif;
    }
    h1 {
        text-align: center;
    }
    .row {
        display: flex;
        flex-direction: row;
    }
    .image-container {
        width: 40%;
        vertical-align: top;
    }
    .text-container {
        width: 60%;
        padding-left: 1rem;
        vertical-align: top;
    }
    .time-container {
        width: 100%;
        display: flex;
        flex-direction: row;
        align-items: center;
        h2 {
            flex-grow: 0;
        }
        hr {
            flex-grow: 1;
            height: 0px;
            margin: 20px;
        }
    }
    .speech-prob-low {
        background-color: #FFAFAF;
    }
    .speech-prob-med {
        background-color: #FFE6AF;
    }
    img {
        max-width: 100%; /* Makes image responsive */
        height: auto;
    }
    .footer {
        background-color: lightgray;
        table {
        table-layout: fixed;
            width: 100%;
            td {
                equal-width: auto;
                text-align: center;
            }
        }
    }
    /* Add more styling as needed */
</style>
</head>
<body>
"""

    if not title:
        title = "Untitled"
    c += f"<h1>{title}</h1>"

    c += """
<div class="disclaimer">
<i>
Do not cite this transcript - instead, verify with the source material.
This is an automatically-generated transcript of a video.
It has not been reviewed for correctness.
It may contain factual errors, distortions, omissions, or entirely ficitious content, including, but not limited to names, dates, places and events -- even entire phrases, sentences, or paragraphs.
</i>
</div>
"""

    c += """
<div class="speech-prob-low" style="border: 1px solid black;">Includes low-probability speech</div>
<div class="speech-prob-med" style="border: 1px solid black;">Includes medium-probability speech</div>
"""

    for (start, stop, speaker_id, text, speech_prob), screencap_ids in segments.items():

        screencaps = [
            session.scalars(select(Screencap).where(Screencap.id == id)).one()
            for id in screencap_ids
        ]

        words = text.strip().split(" ")

        c += f"""
<div class="row">
    <div class="time-container">
        <hr>
        <h2>{hh_mm_ss_ms(start/1000)} - {hh_mm_ss_ms(stop/1000)}</h2>
        <hr>
    </div>
</div>
"""

        # estimate which words should follow each screencap
        for sci, sc in enumerate(screencaps):

            screencap_path = root_dir / "screencaps" / sc.name

            this_frac = (sc.timestamp_ms - start) / (stop - start)
            this_frac = max(0, this_frac)
            this_frac = min(this_frac, 1)
            this_start_index = int(this_frac * len(words))
            if sci == len(screencaps) - 1:
                # last screencap is always followed by all remaining words
                word_range = (this_start_index, len(words))
            else:
                # compute the words at which the next screencap starts
                next_frac = (screencaps[sci + 1].timestamp_ms - start) / (stop - start)
                next_frac = max(0, next_frac)
                next_frac = min(next_frac, 1)
                next_start_index = int(next_frac * len(words))
                word_range = (this_start_index, next_start_index)

            some_words = words[word_range[0] : word_range[1]]
            if speech_prob > 0.8:
                prob = "hi"
            elif speech_prob > 0.4:
                prob = "med"
            else:
                prob = "low"
            c += f"""
<div class="row">
    <div class="image-container">
        <img src="{screencap_path.relative_to(root_dir.resolve())}" alt="Your Image">
    </div>
    <div class="text-container speech-prob-{prob}">
        <p>{' '.join(some_words)}</p>
    </div>
</div>
"""

    c += f"""
<div class="footer">
<table>
    <tr>
    <td>Made by github.com/cwpearson/cabq-video</td>
    <td>Generated {datetime.datetime.now():%a %b %d, %Y %I:%M %p}</td>
    <td>Url</td>
    </tr>
</table>
</div>
"""

    c += "\n</body>\n</html>\n"
    return c


def get_engine(dir: Path):
    db_path = dir / "video.db"
    engine_path = f"sqlite:///{db_path}"
    log(f"open {engine_path}")
    return create_engine(engine_path, echo=False)


def hash_file_data(f) -> str:
    """hash the first 128k bytes of the file"""
    return hashlib.sha256(f.read(128 * 1024)).hexdigest()


def get_screencap(
    session: Session, video_path: Path, timestamp: float, root_dir: Path
) -> int:
    screencap_dir = root_dir / "screencaps"
    screencap_dir.mkdir(exist_ok=True, parents=True)

    timestamp_ms = int(timestamp * 1000 + 0.5)

    screencap = session.scalars(
        select(Screencap).where(Screencap.timestamp_ms == timestamp_ms)
    ).one_or_none()

    if screencap is not None:
        output_path = screencap_dir / screencap.name
        if output_path.is_file():
            return screencap.id
        else:
            # need to regenerate screencap
            session.delete(screencap)
            session.commit()
    try:
        f = NamedTemporaryFile(delete=False, dir=screencap_dir, suffix=".jpg")
        output_path = screencap_dir / f.name
        cmd = [
            "ffmpeg",
            "-y",  # overwrite output
            "-ss",
            str(timestamp),
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            str(output_path),
        ]
        log(f"run {cmd}")
        subprocess.run(cmd, capture_output=True).check_returncode()

        screencap = Screencap(name=f.name, timestamp_ms=timestamp_ms)
        session.add(screencap)
        session.commit()
        return screencap.id
    except Exception as e:
        output_path.unlink()
        raise e


def assign_segments(session: Session):
    """
    assign segments to whichever speakers are most prominently
    featured during those segments
    """

    log(f"assign segments...")
    turns = session.scalars(select(Turn)).all()
    segments = session.scalars(select(Segment).where(Segment.speaker_id == None)).all()

    ncls = NCLS(
        [t.start_ms for t in turns],
        [t.stop_ms for t in turns],
        [t.id for t in turns],
    )

    for s in segments:
        s_start = s.start_ms
        s_stop = s.stop_ms
        amounts = {}
        for t_start, t_stop, t_id in ncls.find_overlap(s_start, s_stop):
            max_start = max(s_start, t_start)
            min_stop = min(s_stop, t_stop)
            amounts[t_id] = amounts.get(t_id, 0) + min_stop - max_start
        if amounts:
            # FIXME: break ties?
            max_turn_id = max(amounts, key=amounts.get)
            turn = session.scalars(select(Turn).where(Turn.id == max_turn_id)).one()
            s.speaker_id = turn.speaker_id
    session.commit()


@click.command()
@click.option("--output-dir", type=Path, default=None)
@click.option("--force-frames", is_flag=True)
@click.option("--force-scenes", is_flag=True)
@click.option("--force-diarize", is_flag=True)
@click.option("--force-transcribe", is_flag=True)
@click.option("--force-chunk", is_flag=True)
@click.option("--title", type=str)
@click.option("--source-url", type=str)
@click.argument("video_path", type=Path)
def main(
    video_path: Path,
    force_frames: bool,
    force_scenes: bool,
    force_diarize: bool,
    force_chunk: bool,
    force_transcribe: bool,
    output_dir: Path = None,
    title: str = None,
    source_url: str = None,
):

    if not title:
        title = video_path.name

    if output_dir is None:
        output_dir = (video_path.parent) / video_path.stem

    if output_dir.is_file():
        raise RuntimeError(f"{output_dir} is a file!")

    log(f"main: create {output_dir}")
    output_dir.mkdir(exist_ok=True, parents=True)

    Base.metadata.create_all(get_engine(output_dir))

    log(f"main: hash video {video_path}")
    with open(video_path, "rb") as f:
        file_hash = hash_file_data(f)
    log(f"main: file hash: {file_hash}")

    with Session(get_engine(output_dir)) as session:

        video = session.scalars(
            select(Video).where(Video.file_hash == file_hash)
        ).one_or_none()

        if not video:
            video = Video(
                original_name=video_path.name,
                file_hash=file_hash,
            )
            session.add(video)
            session.commit()
            log(f"main: created new video")
        else:
            log(f"main: matched existing video")

        if force_frames:
            tr.frames.remove_frames(session)
        if not video.frames_hashed:
            # tr.frames.hash_frames(session, video_path, output_dir)
            tr.frames.hash_frames_mp(session, video_path, output_dir)

        if force_scenes or force_frames:
            remove_scenes(session)
        if not video.scenes_detected:
            detect_scenes(session, video_path, output_dir)

        if force_diarize:
            remove_diarize(session)
        if not video.diarized:
            diarize(session, video_path, output_dir)
        else:
            log(f"main: video already diarized")

        if force_diarize or force_chunk:
            for chunk in session.scalars(select(SpeakerChunk)).all():
                session.delete(chunk)
            session.commit()
        make_speaker_chunks(session)

        if force_diarize or force_chunk or force_transcribe:
            remove_transcribe(session)
        if not video.transcribed:
            transcribe(session, video_path, output_dir)
        else:
            log(f"main: video already transcribed")

        assign_segments(session)

        text_segments = condense_segments(session)

        contents = text_output(session, text_segments)
        output_path = output_dir / "tr.txt"
        log(f"main: write {output_path}")
        with open(output_path, "w") as f:
            f.write(contents)

        # get a screencap at the beginning of each text segment,
        # and every scene change or every 30 seconds
        segment_screencaps = {}
        for s in text_segments:
            start_ms = s[0]
            stop_ms = s[1]

            # all scenes in this segment
            scenes = session.scalars(
                select(Scene)
                .where(Scene.start_ms >= start_ms)
                .where(Scene.stop_ms <= stop_ms)
                .order_by(Scene.start_ms)
            ).all()

            # screencap at beginning
            target_timestamps = [start_ms]
            # screencap at the start of every scene
            target_timestamps += [scene.start_ms for scene in scenes]

            # screencap at the end
            target_timestamps += [stop_ms]
            # additional screencaps during long scenes

            new_timestamps = []
            for ts1, ts2 in zip(target_timestamps[:-1], target_timestamps[1:]):
                num_caps = math.ceil((ts2 - ts1) / 60000)
                for ts in range(ts1, ts2, math.ceil((ts2 - ts1) / num_caps)):
                    new_timestamps += [ts]
            target_timestamps = new_timestamps

            sc_ids = []
            for ts in target_timestamps:
                sc_ids += [get_screencap(session, video_path, ts / 1000, output_dir)]

            # if the first screencap is the same as the first scene change,
            # remove the first scene change
            if len(sc_ids) >= 2:
                sc0 = session.scalars(
                    select(Screencap).where(Screencap.id == sc_ids[0])
                ).one()
                sc1 = session.scalars(
                    select(Screencap).where(Screencap.id == sc_ids[1])
                ).one()
                hash_0 = imagehash.dhash(
                    Image.open(output_dir / "screencaps" / sc0.name)
                )
                hash_1 = imagehash.dhash(
                    Image.open(output_dir / "screencaps" / sc1.name)
                )
                if hash_0 - hash_1 < SCENE_CHANGE_DHASH_DELTA:
                    sc_ids = sc_ids[0:1] + sc_ids[2:]
            segment_screencaps[s] = sc_ids

        contents = html_output(session, segment_screencaps, output_dir, title=title)
        output_path = output_dir / "index.html"
        log(f"main: write {output_path}")
        with open(output_path, "w") as f:
            f.write(contents)


if __name__ == "__main__":
    main()
