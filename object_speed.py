import supervision as sv

import cv2
import json
import torch
import datetime
import itertools
import numpy as np
from pathlib import Path
from collections import defaultdict, deque

from imutils.video import FileVideoStream, WebcamVideoStream

from sinks.model_sink import ModelSink
from sinks.annotation_sink import AnnotationSink

import config
from tools.video_info import VideoInfo
from tools.messages import source_message, progress_message, step_message
from tools.write_data import csv_append, write_csv
from tools.general import load_zones
from tools.speed import ViewTransformer

# For debugging
from icecream import ic


def main(
    source: str = '0',
    output: str = 'output',
    weights: str = 'yolov10b.pt',
    class_filter: list[int] = None,
    image_size: int = 640,
    confidence: int = 0.5,
) -> None:
    # Initialize video source
    source_info, source_flag = VideoInfo.get_source_info(source)
    step_message(next(step_count), 'Video Source Initialized ✅')
    source_message(source, source_info)

    # Check GPU availability
    step_message(next(step_count), f"Processor: {'GPU ✅' if torch.cuda.is_available() else 'CPU ⚠️'}")

    # Initialize YOLOv10 model
    track_sink = ModelSink(
        weights_path=weights,
        image_size=image_size,
        confidence=confidence,
        class_filter=class_filter )
    step_message(next(step_count), f"{Path(weights).stem.upper()} Model Initialized ✅")

    # show_image size
    scaled_width = 1280 if source_info.width > 1280 else source_info.width
    scaled_height = int(scaled_width * source_info.height / source_info.width)
    scaled_height = scaled_height if source_info.height > scaled_height else source_info.height

    # Annotators
    annotation_sink = AnnotationSink(
        source_info=source_info,
        trace=True,
        label=False,
        fps=False
    )

    line_thickness = int(sv.calculate_optimal_line_thickness(resolution_wh=source_info.resolution_wh) * 0.5)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=source_info.resolution_wh) * 0.5
    label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_padding=2, text_position=sv.Position.TOP_LEFT, text_thickness=line_thickness)

    # Regions
    target_width = 730
    target_height = 5000
    with open(f"{config.SOURCE_FOLDER}/{config.JSON_NAME}", 'r') as json_file:
        json_data = json.load(json_file)
        zone_analysis = np.array(json_data[0]).astype(np.int32)
        zone_target = np.array( [ [0, 0], [target_width, 0], [target_width, target_height], [0, target_height] ] )

    polygon_zone = sv.PolygonZone(polygon=zone_analysis, frame_resolution_wh=(source_info.width,source_info.height))
    view_transformer = ViewTransformer(source=zone_analysis, target=zone_target)
    coordinates = defaultdict(lambda: deque(maxlen=int(source_info.fps)))
    speeds = defaultdict(lambda: deque(maxlen=10))

    # Start video tracking processing
    step_message(next(step_count), 'Video Tracking Started ✅')
    
    if source_flag == 'stream':
        video_stream = WebcamVideoStream(src=eval(source) if source.isnumeric() else source)
        source_writer = cv2.VideoWriter(f"{output}_source.mp4", cv2.VideoWriter_fourcc(*'mp4v'), source_info.fps, (source_info.width, source_info.height))
    elif source_flag == 'video':
        video_stream = FileVideoStream(source)
        output_writer = cv2.VideoWriter(f"{output}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), source_info.fps, (source_info.width, source_info.height))

    frame_number = 0
    output_data = []
    video_stream.start()
    time_start = datetime.datetime.now()
    fps_monitor = sv.FPSMonitor()
    try:
        while video_stream.more() if source_flag == 'video' else True:
            fps_monitor.tick()
            fps_value = fps_monitor.fps

            image = video_stream.read()
            if image is None:
                print()
                break

            annotated_image = image.copy()

            # YOLO inference
            results = track_sink.track(image=image)
                
            # Save object data in list
            output_data = csv_append(output_data, frame_number, results)

            # Convert results to Supervision format
            detections = sv.Detections.from_ultralytics(results)


            # Draw annotations
            annotated_image = annotation_sink.on_detections(detections=detections, image=image)


            # Speed
            if len(detections) > 0:
                points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
                points = view_transformer.transform_points(points=points).astype(int)
                object_labels =[]
                if detections.tracker_id is not None:
                    for tracker_id, [x, y] in zip(detections.tracker_id, points):
                        coordinates[tracker_id].append([frame_number, x, y])
                        if len(coordinates[tracker_id]) < source_info.fps / 2:
                            object_labels.append(f"#{tracker_id}")
                        else:
                            t_0, x_0, y_0 = coordinates[tracker_id][0]
                            t_1, x_1, y_1 = coordinates[tracker_id][-1]

                            distance = np.sqrt((y_1-y_0)**2 + (x_1-x_0)**2) / 100
                            time_diff = (t_1 - t_0) / source_info.fps

                            speeds[tracker_id].append(distance / time_diff * 3.6)

                            mean_speed = sum(speeds[tracker_id]) / len(speeds[tracker_id])

                            object_labels.append(f"#{tracker_id} {int(mean_speed)} Km/h")

                    annotated_image = label_annotator.annotate(
                        scene=annotated_image,
                        detections=detections,
                        labels=object_labels )

            # Save results
            output_writer.write(annotated_image)
            if source_flag == 'stream': source_writer.write(image)

            # Print progress
            progress_message(frame_number, source_info.total_frames, fps_value)
            frame_number += 1

            # View live results
            cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Output', int(scaled_width), int(scaled_height))
            cv2.imshow("Output", annotated_image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n")
                break

    except KeyboardInterrupt:
        step_message(next(step_count), 'End of Video ✅')
    step_message(next(step_count), 'Saving Detections in CSV file ✅')
    write_csv(f"{output}.csv", output_data)
    
    step_message(next(step_count), f"Elapsed Time: {(datetime.datetime.now() - time_start).total_seconds():.2f} s")
    output_writer.release()
    if source_flag == 'stream': source_writer.release()
    
    cv2.destroyAllWindows()
    video_stream.stop()


if __name__ == "__main__":
    step_count = itertools.count(1)
    main(
        source=f"{config.SOURCE_FOLDER}/{config.INPUT_VIDEO}",
        output=f"{config.OUTPUT_FOLDER}/{config.OUTPUT_NAME}",
        weights=f"{config.MODEL_FOLDER}/{config.MODEL_WEIGHTS}",
        class_filter=config.CLASS_FILTER,
        image_size=config.IMAGE_SIZE,
        confidence=config.CONFIDENCE,
    )
