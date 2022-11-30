import json
import os.path as osp
import zipfile
from collections import OrderedDict
from glob import glob
import shutil
from tempfile import TemporaryDirectory
from typing import Callable

from cvat.apps.dataset_manager.util import make_zip_archive
from cvat.apps.dataset_manager.bindings import (ProjectData, CommonData,
                                                get_defaulted_subset,
                                                import_dm_annotations,
                                                match_dm_item,TaskData, JobData)
from .registry import exporter, importer

def pairwise(iterable):
    a = iter(iterable)
    return zip(a, a)

def dump_json_file(file_object, annotations):
    # location = {"jangsan"}
    # weather = {"sunny", "rainy", "cloudy"}
    # time = {"morning", "afternoon"}
    action_class = {"fall_down", "sos_wave"}
    action_class_id = {"fall_down":1, "sos_wave":2}
    object_class = {"person", "car"}
    object_class_id = {"person":3, "car":4}

    # print(annotations.meta)
    filename = str(annotations.meta['task']['name'])

    split_text = filename.split("_")
    year_month_day = split_text[0]
    hour_min = split_text[1]
    location = split_text[2]
    degree = split_text[3]
    altitude = split_text[4]
    weather = split_text[5]
    time = split_text[6]
    height = int(annotations.meta['task']['original_size']['height'])
    width = int(annotations.meta['task']['original_size']['width'])

    file_data = OrderedDict()
    file_data["info"] ={
        "file_name": str(filename),
        "height": height,
        "width": width,
        "date": str(year_month_day + "_" + hour_min),
        "location": str(location),
        "time": str(time),
        "weather": str(weather),
        "fps": "24",
        "altitude": int(altitude),
        "degree": int(degree),
        # "tracks": str(annotations.tracks)
    }

    file_data["action_events"] = []
    file_data["action_annotation"] = []
    file_data["object_annotation"] = []

    frame_data = OrderedDict()
    frame_data["frame_bbox"] = []

    track_id = 0
    action_flag = False
    object_flag = False
    action_max_frame = 0
    action_person_id = None
    attr_name = None

    compare_id = 0

    ## TEST tracks
    # for track in annotations.tracks:
    #     for shape in track.shapes:
    #         file_data["action_events"].append(shape.label)
    #

    # Create Track(of cvat function) Action Annotation
    for track in annotations.tracks:
        track_id += 1
        action_events_label = []
        action_annotation_label = []
        action_annotation_coordinate =[]

        for shape in track.shapes:
            # Action
            if shape.label in action_class:
                action_flag = True
                action_id = action_class_id[shape.label]

            if action_flag:
                for attr in shape.attributes:
                    attr_name = attr.name
                    action_person_id = attr.value

                # Divide coordinates by action
                # coordinates pick up from shape
                if len(action_annotation_label) == 0:
                    action_annotation_label.append([action_id, action_person_id, track_id, shape.label,
                                                    action_annotation_coordinate])
                    action_annotation_label[compare_id][4].append([shape.frame, shape.points[0], shape.points[1],
                                                         shape.points[2], shape.points[3]])
                else:
                    if action_id == action_annotation_label[compare_id][0] and \
                        action_person_id == action_annotation_label[compare_id][1]:
                        # action_annotation_coordinate.append([shape.frame, shape.points[0], shape.points[1],
                        #                                      shape.points[2], shape.points[3]])
                        action_annotation_label[compare_id][4].append([shape.frame, shape.points[0], shape.points[1],
                                                             shape.points[2], shape.points[3]])
                    else:
                        action_annotation_label.append([action_id, action_person_id, track_id, shape.label,
                                                        action_annotation_coordinate])
                        action_annotation_label[compare_id+1][4].append([action_id, action_person_id, track_id, shape.label,
                                                        action_annotation_coordinate])
                        compare_id += 1

                # events pick up from shape
                if attr_name == "action_person_id":
                    action_min_frame = shape.frame
                    if len(action_events_label) == 0:
                        action_events_label.append([action_person_id, action_id, action_min_frame, action_max_frame,
                                                    track_id, shape.label])
                    else:
                        for i in range(len(action_events_label)):
                            if action_events_label[i][4] == track_id:
                                action_events_label[i][3] = shape.frame
                            else:
                                action_events_label.append([action_person_id, action_id, action_min_frame,
                                                            action_max_frame, track_id, shape.label])
            action_flag = False


        # Write json for action annotation parts
        if len(action_annotation_label) > 0:
            for i in range(len(action_annotation_label)):
                for i2 in range(len(action_annotation_label[i][4])):
                    frame_data["frame_bbox"].append({
                        "bbox_frame": int(action_annotation_label[i][4][i2][0]),
                        "bbox": [
                            float(action_annotation_label[i][4][i2][0]),
                            float(action_annotation_label[i][4][i2][1]),
                            float(action_annotation_label[i][4][i2][2]),
                            float(action_annotation_label[i][4][i2][3])
                        ]
                    })
                file_data["action_annotation"].append({
                    "action_id": int(action_annotation_label[i][0]),
                    "action_name": str(action_annotation_label[i][3]),
                    "action_person_id": int(action_annotation_label[i][2]),
                    "frame_bbox": frame_data["frame_bbox"]
                })

        # Write json for action event parts
        if len(action_events_label) > 0:
            for i in range(len(action_events_label)):
                file_data["action_events"].append({
                    "action_id": int(action_events_label[i][1]),
                    "action_start_frame": int(action_events_label[i][2]),
                    "action_end_frame": int(action_events_label[i][3]),
                    "action_name": str(action_events_label[i][5]),
                    "action_person_id": int(action_events_label[i][0])
                })

    # Create Frame By Frame Object Annotation
    for frame_annotation in annotations.group_by_frame(include_empty=True):
        for shape in frame_annotation.labeled_shapes:
            object_annotation_label = []

            if shape.label in object_class:
                object_flag = True
                object_id = object_class_id[shape.label]

                if object_flag:
                    object_annotation_label.append([object_id, shape.frame, shape.label, [shape.points[0], shape.points[1],
                                                                                          shape.points[2], shape.points[3]]])
                object_flag = False

            # Write json for object parts
            if len(object_annotation_label) > 0:
                for i in range(len(object_annotation_label)):
                    file_data["object_annotation"].append({
                        "object_id": int(object_annotation_label[i][0]),
                        "cur_frame": int(object_annotation_label[i][1]),
                        "object_name": str(object_annotation_label[i][2]),
                        "bbox": [
                            float(object_annotation_label[i][3][0]),
                            float(object_annotation_label[i][3][1]),
                            float(object_annotation_label[i][3][2]),
                            float(object_annotation_label[i][3][3]),
                        ]
                    })

    json.dump(file_data, file_object, ensure_ascii=False, indent='\t')

def dump_json_file_project(temp_dir, project_data):
    # location = {"jangsan"}
    # weather = {"sunny", "rainy", "cloudy"}
    # time = {"morning", "afternoon"}
    action_class = {"fall_down", "sos_wave"}
    action_class_id = {"fall_down":1, "sos_wave":2}
    object_class = {"person", "car"}
    object_class_id = {"person":3, "car":4}

    for i in range(len(project_data.meta['project']['tasks'])):
        file_name = project_data.meta['project']['tasks'][i][1]['name']
        task_id = project_data.meta['project']['tasks'][i][1]['id']
        split_text = file_name.split("_")
        year_month_day = split_text[0]
        hour_min = split_text[1]
        location = split_text[2]
        degree = split_text[3]
        altitude = split_text[4]
        weather = split_text[5]
        time = split_text[6]

        track_id = 0
        action_flag = False
        object_flag = False
        action_max_frame = 0
        action_person_id = None
        attr_name = None

        compare_id = 0

        file_object = open(osp.join(temp_dir, file_name + ".json"), 'w', encoding='utf-8')
        file_data = OrderedDict()
        file_data["info"] = {
            "file_name": str(file_name),
            "height": int(project_data.meta['project']['tasks'][i][1]['original_size']['height']),
            "width": int(project_data.meta['project']['tasks'][i][1]['original_size']['width']),
            "date": str(year_month_day + "_" + hour_min),
            "location": str(location),
            "time": str(time),
            "weather": str(weather),
            "fps": str("24"),
            "altitude": int(altitude),
            "degree": int(degree)
            # "file_name": str(project_data.tracks)

            }
        file_data["action_events"] = []
        file_data["action_annotation"] = []
        file_data["object_annotation"] = []
        frame_data = OrderedDict()
        frame_data["frame_bbox"] = []

        for track in project_data.tracks:
            track_id += 1
            action_events_label = []
            action_annotation_label = []
            action_annotation_coordinate =[]

            if track.task_id == int(task_id):
                for shape in track.shapes:
                    if shape.label in action_class:
                        action_flag = True
                        action_id = action_class_id[shape.label]

                    if action_flag:
                        for attr in shape.attributes:
                            attr_name = attr.name
                            action_person_id = attr.value

                        # Divide coordinates by action
                        # coordinates pick up from shape
                        if len(action_annotation_label) == 0:
                            action_annotation_label.append([action_id, action_person_id, track_id, shape.label,
                                                            action_annotation_coordinate])
                            action_annotation_label[compare_id][4].append([shape.frame, shape.points[0], shape.points[1],
                                                                           shape.points[2], shape.points[3]])
                        else:
                            if action_id == action_annotation_label[compare_id][0] and \
                                action_person_id == action_annotation_label[compare_id][1]:
                                action_annotation_label[compare_id][4].append([shape.frame, shape.points[0],
                                                                               shape.points[1], shape.points[2],
                                                                               shape.points[3]])
                            else:
                                action_annotation_label.append([action_id, action_person_id, track_id, shape.label,
                                                                action_annotation_coordinate])
                                action_annotation_label[compare_id+1][4].append([action_id, action_person_id, track_id,
                                                                                 shape.label, action_annotation_coordinate])
                                compare_id += 1

                                    # events pick up from shape
                        if attr_name == "action_person_id":
                            action_min_frame = shape.frame
                            if len(action_events_label) == 0:
                                action_events_label.append([action_person_id, action_id, action_min_frame,
                                                            action_max_frame, track_id, shape.label])
                            else:
                                for i in range(len(action_events_label)):
                                    if action_events_label[i][4] == track_id:
                                        action_events_label[i][3] = shape.frame
                                    else:
                                        action_events_label.append([action_person_id, action_id, action_min_frame,
                                                                    action_max_frame, track_id, shape.label])
                    action_flag = False

            # write action export json
            if len(action_annotation_label) > 0:
                for i in range(len(action_annotation_label)):
                    for i2 in range(len(action_annotation_label[i][4])):
                        frame_data["frame_bbox"].append({
                            "bbox_frame": int(action_annotation_label[i][4][i2][0]),
                            "bbox": [
                                float(action_annotation_label[i][4][i2][1]),
                                float(action_annotation_label[i][4][i2][2]),
                                float(action_annotation_label[i][4][i2][3]),
                                float(action_annotation_label[i][4][i2][4])
                            ]
                        })
                    file_data["action_annotation"].append({
                        "action_id": int(action_annotation_label[i][0]),
                        "action_name": str(action_annotation_label[i][3]),
                        "action_person_id": int(action_annotation_label[i][2]),
                        "frame_bbox": frame_data["frame_bbox"]
                    })

            # Write json for action event parts
            if len(action_events_label) > 0:
                for i in range(len(action_events_label)):
                    file_data["action_events"].append({
                        "action_id": int(action_events_label[i][1]),
                        "action_start_frame": int(action_events_label[i][2]),
                        "action_end_frame": int(action_events_label[i][3]),
                        "action_name": str(action_events_label[i][5]),
                        "action_person_id": int(action_events_label[i][0])
                    })

        # Create Frame By Frame Object Annotation
        for frame_annotation in project_data.group_by_frame(include_empty=False):

            if frame_annotation.task_id == int(task_id):
                for shape in frame_annotation.labeled_shapes:
                    object_annotation_label = []
                    if shape.label in object_class:
                        object_flag = True
                        object_id = object_class_id[shape.label]

                        if object_flag:
                            object_annotation_label.append([object_id, shape.frame, shape.label,
                                                            [shape.points[0], shape.points[1],
                                                             shape.points[2], shape.points[3]]])
                    if len(object_annotation_label) > 0:
                        for i in range(len(object_annotation_label)):
                            file_data["object_annotation"].append({
                                "object_id": int(object_annotation_label[i][0]),
                                "cur_frame": int(object_annotation_label[i][1]),
                                "object_name": str(object_annotation_label[i][2]),
                                "bbox": [
                                    float(object_annotation_label[i][3][0]),
                                    float(object_annotation_label[i][3][1]),
                                    float(object_annotation_label[i][3][2]),
                                    float(object_annotation_label[i][3][3]),
                                ]
                            })

        json.dump(file_data, file_object, ensure_ascii=False, indent='\t')
        file_object.close()


def _export_task_or_job(dst_file, instance_data, anno_callback, save_images=False):
    with TemporaryDirectory() as temp_dir:
        with open(osp.join(temp_dir,  'annotations_.json'), 'w', encoding='utf-8') as f:
            anno_callback(f, instance_data)
        shutil.move(osp.join(temp_dir, 'annotations_.json'), dst_file.name)


def _export_project(dst_file: str, project_data: ProjectData, anno_callback: Callable, save_images: bool=False):
    # filename, ext = osp.splitext(TaskData.META_FIELD["name"])
    with TemporaryDirectory() as temp_dir:
        anno_callback(temp_dir, project_data)

        make_zip_archive(temp_dir, dst_file)

@exporter(name='drone_export_track_shape', ext='.json, .zip', version='1.0')
def _export_video(dst_file, instance_data, save_images=False):
    if isinstance(instance_data, ProjectData):
        _export_project(dst_file, instance_data, anno_callback=dump_json_file_project, save_images=save_images)
    else:
        _export_task_or_job(dst_file, instance_data, anno_callback=dump_json_file, save_images=save_images)
