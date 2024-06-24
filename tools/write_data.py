from ultralytics.engine.results import Results

import csv
import numpy as np


def csv_append(output: list, frame_number: int, ultralytics_results: Results) -> list:
    """
    Appends the results of the Ultralytics object detection to the output list.

    Parameters:
        output (list): The list to which the results are appended.
        frame_number (int): The frame number of the video from which the objects are detected.
        ultralytics_results (ultralytics.engine.results.Results): The results from Ultralytics 
            object detection.

    Returns:
        output (list): The list to which the results are appended.
            Each result is a list containing the frame number, tracker id, class name, 
            bounding box coordinates (x, y, width, height), and confidence score.
    """
    xyxys = ultralytics_results.boxes.xyxy.cpu().numpy()
    confidences = ultralytics_results.boxes.conf.cpu().numpy()
    tracker_ids=ultralytics_results.boxes.id.int().cpu().numpy() \
        if ultralytics_results.boxes.id is not None \
        else np.empty(len(xyxys), dtype=object)
    class_ids = ultralytics_results.boxes.cls.cpu().numpy().astype(int)
    class_names = np.array([ultralytics_results.names[i] for i in class_ids])

    for xyxy, confidence, tracker_id, class_name in zip(xyxys, confidences, tracker_ids, class_names):
        x = xyxy[0]
        y = xyxy[1]
        w = xyxy[2]-xyxy[0]
        h = xyxy[3]-xyxy[1]
        output.append([frame_number, tracker_id, class_name, x, y, w, h, confidence])

    return output


def write_csv(save_path: str, data: list) -> None:
    """
    Writes a list of data into a CSV file.

    Parameters:
        save_path (str): The path where the CSV file will be saved.
        data (list): The data to be written into the CSV file. Each element of the list is 
            expected to be a list itself, representing a row in the CSV.

    Returns:
        None
    """
    with open(save_path, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(data)


def txt_append(output: list, ultralytics_results: Results) -> list:
    """
    Appends the results of the Ultralytics object detection to the output list in a format suitable for a text file.

    Parameters:
        output (list): The list to which the results are appended.
        ultralytics_results (ultralytics.engine.results.Results): The results from Ultralytics 
            object detection.

    Returns:
        output: The output list with appended results. 
            Each result is a list containing the class id and the normalized bounding box 
            coordinates (center x, center y, width, height).
    """
    xywhns = ultralytics_results.boxes.xywhn.cpu().numpy()
    class_ids = ultralytics_results.boxes.cls.cpu().numpy().astype(int)

    for class_id, xywhn in zip(class_ids, xywhns):
        x = xywhn[0]
        y = xywhn[1]
        w = xywhn[2]
        h = xywhn[3]
        output.append([class_id, x, y, w, h])

    return output


def write_txt(save_path: str, data: list) -> None:
    """
    Writes a list of data into a TXT file with labelling format.

    Parameters:
        save_path (str): The path where the TXT file will be saved.
        data (list): The data to be written into the TXT file. Each element of the list is 
            expected to be a list itself, representing a row in the TXT.

    Returns:
        None
    """
    with open(save_path, 'a', newline='') as txt_file:
        txt_writer = csv.writer(txt_file, delimiter=' ')
        txt_writer.writerows(data)
        