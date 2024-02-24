from pathlib import Path
import multiprocessing
import queue

from sqlalchemy.orm import Mapped, mapped_column, Session
from sqlalchemy import select, Text, Integer, desc, create_engine

import cv2
from tqdm import tqdm
from PIL import Image
import imagehash

from transcript.video import Base, Video
from transcript.logging import log

FRAME_INTERVAL = 0.5


class Frame(Base):
    __tablename__ = "frame"
    id: Mapped[int] = mapped_column(primary_key=True)
    target_ms: Mapped[int] = mapped_column(Integer)
    actual_ms: Mapped[int] = mapped_column(Integer)
    dhash: Mapped[str] = mapped_column(Text)


def convert_cv2_to_pil(cv2_img):
    # w, h = 196, 196
    # cv2_img = cv2.resize(cv2_img, (w, h), interpolation=cv2.INTER_AREA)
    cv2_img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cv2_img_rgb)
    return pil_img


def get_engine(dir: Path):
    db_path = dir / "video.db"
    engine_path = f"sqlite:///{db_path}"
    # log(f"open {engine_path}")
    return create_engine(engine_path, echo=False)


def hash_frames_worker(
    video_path: Path, root_dir: Path, to_hash: multiprocessing.Queue
):
    with Session(get_engine(root_dir)) as session:

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            log(f"detect_scenes: Error opening video file")
            return
        else:
            while True:
                target_timestamp = to_hash.get()
                if "done" == target_timestamp:
                    print(f"worker exiting")
                    cap.release()
                    return

                cap.set(cv2.CAP_PROP_POS_MSEC, target_timestamp)
                ret, frame = cap.read()
                if not ret:
                    cap.release()
                    return
                actual_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
                pil_image = convert_cv2_to_pil(frame)
                dhash = imagehash.dhash(pil_image)
                session.add(
                    Frame(
                        target_ms=int(target_timestamp),
                        actual_ms=int(actual_timestamp),
                        dhash=str(dhash),
                    )
                )
                session.commit()


def hash_frames_mp(session: Session, video_path, root_dir: Path):

    n_workers = max(1, int(multiprocessing.cpu_count() / 3 + 0.5))
    log(f"hash_frames_mp: using {n_workers} workers")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError()

    # length in ms
    video_length = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS) * 1000
    log(f"hash_frames_mp: video length {video_length:.1f}ms")
    num_frames = int(video_length // (FRAME_INTERVAL * 1000))
    log(
        f"hash_frames_mp: will take {num_frames} frames (interval is {FRAME_INTERVAL}s)"
    )

    to_hash = multiprocessing.Queue(maxsize=1)

    workers = [
        multiprocessing.Process(
            target=hash_frames_worker,
            args=(
                video_path,
                root_dir,
                to_hash,
            ),
        )
        for _ in range(n_workers)
    ]
    for w in workers:
        w.start()

    pbar = tqdm(total=num_frames, unit="frames")  # progress bar in seconds

    target_timestamp = 0
    while target_timestamp <= video_length:

        existing = session.scalars(
            select(Frame).where(Frame.target_ms == target_timestamp)
        ).one_or_none()
        if existing is None:
            to_hash.put(target_timestamp)
            pbar.n += 1
        else:
            pbar.total -= 1
        pbar.refresh()
        target_timestamp = target_timestamp + FRAME_INTERVAL * 1000

    for w in workers:
        to_hash.put("done")
    for w in workers:
        w.join()

    session.scalars(select(Video)).one().frames_hashed = True
    session.commit()


def hash_frames(session: Session, video_path, root_dir: Path):
    video = session.scalars(select(Video)).one()
    cap = cv2.VideoCapture(str(video_path))

    # Check if the video was opened successfully
    if not cap.isOpened():
        log(f"detect_scenes: Error opening video file")
    else:
        # length in ms
        video_length = (
            cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS) * 1000
        )

        existing = session.scalars(
            select(Frame).order_by(desc(Frame.target_ms)).limit(1)
        ).one_or_none()

        if existing is not None:
            target_timestamp = existing.target_ms
        else:
            target_timestamp = 0

        pbar = tqdm(total=int(video_length / 1000))  # progress bar in seconds
        while target_timestamp <= video_length:

            cap.set(cv2.CAP_PROP_POS_MSEC, target_timestamp)

            ret, frame = cap.read()
            # If frame is read correctly ret is True
            if not ret:
                break

            actual_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)

            # print(f"{actual_timestamp / 1000:.1f}")

            # Convert the frame to PIL Image
            pil_image = convert_cv2_to_pil(frame)
            dhash = imagehash.dhash(pil_image)
            session.add(
                Frame(
                    target_ms=int(target_timestamp),
                    actual_ms=int(actual_timestamp),
                    dhash=str(dhash),
                )
            )
            session.commit()

            pbar.n = int(actual_timestamp / 1000)  # progress bar in seconds
            pbar.refresh()

            target_timestamp = target_timestamp + FRAME_INTERVAL * 1000

    # When everything done, release the capture
    cap.release()
    video.frames_hashed = True
    session.commit()


def remove_frames(session: Session):
    video = session.scalars(select(Video)).one()
    frames = session.scalars(select(Frame)).all()
    for frame in frames:
        session.delete(frame)
    video.frames_hashed = False
    session.commit()
