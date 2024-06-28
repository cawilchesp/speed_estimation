import supervision as sv

import cv2
import json
import numpy as np
from collections import defaultdict, deque


class SpeedSink:
    def __init__(
        self,
        real_width: float = None,
        real_length: float = None,
        region_json: str = None,
        source_info = None,
        draw_zone: bool = False
    ) -> None:
        with open(region_json, 'r') as json_file:
            json_data = json.load(json_file)
            self.zone_analysis = np.array(json_data[0]).astype(np.int32)
        self.zone_target = np.array( [ [0, 0], [real_width, 0], [real_width, real_length], [0, real_length] ] )
        source = self.zone_analysis.astype(np.float32)
        target = self.zone_target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

        self.source_info = source_info
        self.coordinates = defaultdict(lambda: deque(maxlen=int(source_info.fps)))
        self.speeds = defaultdict(lambda: deque(maxlen=10))

        line_thickness = int(sv.calculate_optimal_line_thickness(resolution_wh=source_info.resolution_wh) * 0.5)
        text_scale = sv.calculate_optimal_text_scale(resolution_wh=source_info.resolution_wh) * 0.5
        self.label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_padding=2, text_position=sv.Position.TOP_LEFT, text_thickness=line_thickness)
        self.draw_zone = draw_zone


    def transform_points(self, detections: sv.Detections) -> np.ndarray:
        if detections.tracker_id is not None:
            points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
            transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
            transformed_points_reshaped = transformed_points.reshape(-1, 2).astype(int)
            
            return transformed_points_reshaped
    

    def speed_estimation(self, detections: sv.Detections, points, frame_number: int):
        if detections.tracker_id is not None:
            object_labels = []
            for tracker_id, [x, y] in zip(detections.tracker_id, points):
                self.coordinates[tracker_id].append([frame_number, x, y])
                if len(self.coordinates[tracker_id]) > self.source_info.fps / 2:
                    t_0, x_0, y_0 = self.coordinates[tracker_id][0]
                    t_1, x_1, y_1 = self.coordinates[tracker_id][-1]

                    distance = np.sqrt((y_1-y_0)**2 + (x_1-x_0)**2) / 100
                    time_diff = (t_1 - t_0) / self.source_info.fps

                    self.speeds[tracker_id].append(distance / time_diff * 3.6)
                    mean_speed = sum(self.speeds[tracker_id]) / len(self.speeds[tracker_id])
                    object_labels.append(f"#{tracker_id} {int(mean_speed)} Km/h")
                else:
                    object_labels.append(f"#{tracker_id}")

            return object_labels


    def speed_annotation(self, detections: sv.Detections, scene: np.array, object_labels: list):
        if self.draw_zone:
            scene = sv.draw_polygon(
                scene=scene,
                polygon=self.zone_analysis,
                color=sv.Color.RED,
                thickness=2,
            )

        if detections.tracker_id is not None:
            scene = self.label_annotator.annotate(
                scene=scene,
                detections=detections,
                labels=object_labels )
        
        return scene