import cv2
from typing import Tuple

class VideoInfo:
    def __init__(
        self,
        width: int = 0,
        height: int = 0,
        fps: int = 0,
        total_frames: float = None
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.total_frames = total_frames

    @property
    def resolution_wh(self) -> Tuple[int, int]:
        return self.width, self.height


    def get_source_info(source: str):
        video_source = eval(source) if source.isnumeric() else source
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened(): raise Exception('Source video not available ❌')
        if source.isnumeric() or source.lower().startswith('rtsp://'):
            source_info = from_camera(cap)
            source_flag = 'stream'
        else:
            source_info = from_video_path(cap)
            source_flag = 'video'
        cap.release()
        
        return source_info, source_flag


def from_video_path(source_cap: cv2.VideoCapture) -> VideoInfo:
    if source_cap.isOpened():
        width = int(source_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(source_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = source_cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(source_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    return VideoInfo(width, height, fps, total_frames)


def from_camera(source_cap: cv2.VideoCapture) -> VideoInfo:
    if source_cap.isOpened():
        width = int(source_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(source_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = source_cap.get(cv2.CAP_PROP_FPS)

    return VideoInfo(width, height, fps)