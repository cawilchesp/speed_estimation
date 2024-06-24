from ultralytics import YOLO
from ultralytics.engine.results import Results

import torch
import numpy as np
from typing import List


class ModelSink:
    def __init__(
        self,
        weights_path: str,
        image_size: int = 640,
        confidence: float = 0.5,
        class_filter: List[int] = None
    ) -> None:
        self.model = YOLO(weights_path)
        self.image_size = image_size        
        self.confidence = confidence
        self.class_filter = class_filter


    def detect(self, image: np.array) -> Results:
        ultralytics_results = self.model(
            source=image,
            imgsz=self.image_size,
            conf=self.confidence,
            classes=self.class_filter,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            verbose=False,
        )[0]
        
        return ultralytics_results
    

    def track(self, image: np.array) -> Results:
        ultralytics_results = self.model.track(
            source=image,
            persist=True,
            imgsz=self.image_size,
            conf=self.confidence,
            classes=self.class_filter,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            verbose=False,
        )[0]
        
        return ultralytics_results
    