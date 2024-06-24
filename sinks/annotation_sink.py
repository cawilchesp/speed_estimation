import supervision as sv

import cv2
import numpy as np

from tools.video_info import VideoInfo


class AnnotationSink:
    def __init__(
        self,
        source_info: VideoInfo,
        fps: bool = True,
        label: bool = True,
        box: bool = True,
        trace: bool = False,
        colorbox: bool = False,
        vertex: bool = True,
        edge: bool = True,
        vertex_label: bool = False,
        track_length: int = 50,
        color_bg: sv.Color = sv.Color(r=0, g=255, b=0),
        color_opacity: float = 0.5,
    ) -> None:
        self.fps = fps
        self.label = label
        self.box = box
        self.trace = trace
        self.colorbox = colorbox
        self.vertex = vertex
        self.edge = edge
        self.vertex_label = vertex_label
        
        # Annotators
        line_thickness = int(sv.calculate_optimal_line_thickness(resolution_wh=source_info.resolution_wh) * 0.5)
        text_scale = sv.calculate_optimal_text_scale(resolution_wh=source_info.resolution_wh) * 0.5
        
        if self.fps: self.fps_monitor = sv.FPSMonitor()
        
        if self.label: self.label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_padding=2, text_position=sv.Position.TOP_LEFT, text_thickness=line_thickness)
        if self.box: self.bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=line_thickness)
        if self.trace: self.trace_annotator = sv.TraceAnnotator(position=sv.Position.CENTER, trace_length=track_length, thickness=line_thickness)
        if self.colorbox: self.color_annotator = sv.ColorAnnotator(color=color_bg, opacity=color_opacity)
        
        if self.vertex: self.vertex_annotator = sv.VertexAnnotator(radius=line_thickness * 3, color=sv.Color.YELLOW)
        if self.edge: self.edge_annotator = sv.EdgeAnnotator(thickness=line_thickness, color=sv.Color.YELLOW)
        if self.vertex_label: self.vertex_label_annotator = sv.VertexLabelAnnotator(border_radius=line_thickness, color=sv.Color.YELLOW, text_color=sv.Color.BLACK)

    def on_detections(self, detections: sv.Detections, scene: np.array) -> np.array:
        # Draw FPS box
        if self.fps:
            self.fps_monitor.tick()
            fps_value = self.fps_monitor.fps

            scene = sv.draw_text(
                scene=scene,
                text=f"{fps_value:.1f}",
                text_anchor=sv.Point(40, 30),
                background_color=sv.Color.from_hex("#A351FB"),
                text_color=sv.Color.from_hex("#000000"),
            )

        # Draw labels
        if self.label:
            if detections.tracker_id is None:
                object_labels = [
                    f"{data['class_name']} ({score:.2f})"
                    for _, _, score, _, _, data in detections
                ]
            else:
                object_labels = [
                    f"{data['class_name']} {tracker_id} ({score:.2f})"
                    for _, _, score, _, tracker_id, data in detections
                ]
            scene = self.label_annotator.annotate(
                scene=scene,
                detections=detections,
                labels=object_labels )
            
        # Draw boxes
        if self.box:
            scene = self.bounding_box_annotator.annotate(
                scene=scene,
                detections=detections )
            
        # Draw tracks
        if self.trace and detections.tracker_id is not None:
            scene = self.trace_annotator.annotate(
                scene=scene,
                detections=detections )
            
        # Draw color boxes
        if self.colorbox:
            scene = self.color_annotator.annotate(
                scene=scene,
                detections=detections )

        return scene
    
    def on_keypoints(self, key_points: sv.KeyPoints, scene: np.array):
        # Draw keypoint vertex
        if self.vertex:
            scene = self.vertex_annotator.annotate(
                scene=scene,
                key_points=key_points )

        # Draw keypoint edges
        if self.edge:
            scene = self.edge_annotator.annotate(
                scene=scene,
                key_points=key_points )

        # Draw keypoint vertex labels
        if self.vertex_label:
            scene = self.vertex_label_annotator.annotate(
                scene=scene,
                key_points=key_points )
        
        return scene