"""Sample torchvision object detection dataset for Boxy

Pretty much just following the pytorch tutorial at
https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

Some code to get started. That's all.
"""
import warnings
import json
import os
import cv2
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from cap.registry import OBJECT_REGISTRY
from typing import Dict, List, Optional, Callable
from cap.data.transforms.detection import ToTensor
from cap.data.transforms.common import PILToTensor
from cap.visualize.bbox2d import vis_det_boxes_2d_waic
from capbc.utils import _as_list
from random import shuffle


@OBJECT_REGISTRY.register
class WaicBoxyDataset(data.Dataset):
    def __init__(self, img_path, label_file,
                 transforms: Optional[List] = None,
                 b_vis=False):
        self.img_path = img_path
        if transforms is not None:
            self.transforms = _as_list(transforms)
        else:
            self.transforms = transforms
        self.labels = self.read_label_file(label_file)
        self.b_vis = b_vis

        self.id_path_map = {}
        counter = 0
        for path, label in self.labels.items():
            if label["vehicles"]:  # vehicles in image
                self.id_path_map[counter] = path
                counter += 1

    def read_label_file(self, label_path: str, min_height: float = 1.0, min_width=1.0):
        """ Reads label file and returns path: label dict
        Args:
            label_path: path to label file (json)
            min_height: minimum AABB height for filtering labels
            min_width: minimum AABB width for filtering labels

        You can't believe how noisy some human annotations are. That single pixel
        width and height filter are in there for a reason
        """
        with open(label_path) as lph:
            labels = json.load(lph)

        for key, label in labels.items():
            pop_list = []
            for vehicle_id, vehicle in enumerate(label["vehicles"]):
                aabb = vehicle["AABB"]
                if aabb["x2"] - aabb["x1"] < min_width or aabb["y2"] - aabb["y1"] < min_height:
                    pop_list.append(vehicle_id)
            for pop_id in reversed(pop_list):
                del label["vehicles"][pop_id]

        return labels

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.img_path, self.id_path_map[idx])
        img = Image.open(img_path).convert("RGB")
        label = self.labels[self.id_path_map[idx]]

        color_space = "rgb"

        # get bounding box coordinates for each mask
        num_objs = len(label["vehicles"])
        boxes = []
        area = torch.FloatTensor([])
        for vehicle in label["vehicles"]:
            aabb = vehicle["AABB"]
            if img.size != (2464, 2056):  # Should use image constants instead
                aabb["x1"] = aabb["x1"] / 2464 * img.size[0]
                aabb["x2"] = aabb["x2"] / 2464 * img.size[0]
                aabb["y1"] = aabb["y1"] / 2056 * img.size[1]
                aabb["y2"] = aabb["y2"] / 2056 * img.size[1]

            aabb["x1"] = aabb["x1"] / img.size[0]
            aabb["x2"] = aabb["x2"] / img.size[0]
            aabb["y1"] = aabb["y1"] / img.size[1]
            aabb["y2"] = aabb["y2"] / img.size[1]

            boxes.append([aabb["x1"], aabb["y1"], aabb["x2"], aabb["y2"], 1])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        image_id = torch.tensor([idx])
        labels = torch.ones((num_objs,), dtype=torch.int64)  # only one class
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)  # to be used if zero

        ig_regions = []
        ig_regions = (
            np.array(ig_regions, dtype=np.float32)
            if len(ig_regions) > 0
            else np.zeros((0, 5), dtype=np.float32)
        )

        boxes = (
            np.array(boxes, dtype=np.float32)
            if len(boxes) > 0
            else np.zeros((0, 5), dtype=np.float32)
        )

        data = {}
        img = np.array(img)
        data["img"] = img
        data["gt_boxes"] = boxes 
        data["ig_regions"] = ig_regions


        if self.transforms is not None:
            for transform in self.transforms:
                data = transform(data)

        if self.b_vis:
            config_vis = dict(
                color=(0, 255, 0),
                thickness=2,
            )
            img_draw = vis_det_boxes_2d_waic(
                vis_image=data["img"], det_boxes_2d=data["gt_boxes"], vis_configs=config_vis)
            cv2.imwrite(os.path.join("/hdd/xfwang_data/vis", str(idx) + '_vehicle.png'), img_draw)


        return data

    def __len__(self):
        return len(self.id_path_map)

# Multi-class segmentation colors for the individual lanes
# The names are based on the camera location, e.g. the markers
# from r2 divide the first lane to the right from the second to the right
DCOLORS = [(110, 30, 30), (75, 25, 230), (75, 180, 60), (200, 130, 0), (48, 130, 245), (180, 30, 145),
           (0, 0, 255), (24, 140, 34), (255, 0, 0), (0, 255, 255),  # the main ones
           (40, 110, 170), (200, 250, 255), (255, 190, 230), (0, 0, 128), (195, 255, 170),
           (0, 128, 128), (195, 255, 170), (75, 25, 230)]
LANE_NAMES = ['l7', 'l6', 'l5', 'l4', 'l3', 'l2',
              'l1', 'l0', 'r0', 'r1',
              'r2', 'r3', 'r4', 'r5',
              'r6', 'r7', 'r8']
DICT_COLORS = dict(zip(LANE_NAMES, DCOLORS))

@OBJECT_REGISTRY.register
class WaicLaneDataset(data.Dataset):
    def __init__(self, img_path, input_folder,
                 transforms: Optional[List] = None,
                 b_vis=False):
        self.img_path = img_path
        if transforms is not None:
            self.transforms = _as_list(transforms)
        else:
            self.transforms = transforms
        self.label_paths = self.get_files_from_folder(input_folder, '.json')
        self.b_vis = b_vis
        shuffle(self.label_paths)

    def filter_lanes_by_size(self, label, min_height=40):
        """ May need some tuning """
        filtered_lanes = []
        for lane in label['lanes']:
            lane_start = min([int(marker['pixel_start']['y']) for marker in lane['markers']])
            lane_end = max([int(marker['pixel_start']['y']) for marker in lane['markers']])
            if (lane_end - lane_start) < min_height:
                continue
            filtered_lanes.append(lane)
        label['lanes'] = filtered_lanes

    def filter_few_markers(self, label, min_markers=2):
        """Filter lines that consist of only few markers"""
        filtered_lanes = []
        for lane in label['lanes']:
            if len(lane['markers']) >= min_markers:
                filtered_lanes.append(lane)
        label['lanes'] = filtered_lanes

    def fix_lane_names(self, label):
        """ Given keys ['l3', 'l2', 'l0', 'r0', 'r2'] returns ['l2', 'l1', 'l0', 'r0', 'r1']"""

        # Create mapping
        l_counter = 0
        r_counter = 0
        mapping = {}
        lane_ids = [lane['lane_id'] for lane in label['lanes']]
        for key in sorted(lane_ids):
            if key[0] == 'l':
                mapping[key] = 'l' + str(l_counter)
                l_counter += 1
            if key[0] == 'r':
                mapping[key] = 'r' + str(r_counter)
                r_counter += 1
        for lane in label['lanes']:
            lane['lane_id'] = mapping[lane['lane_id']]

    def read_json(self, json_path, min_lane_height=20):
        """ Reads and cleans label file information by path"""
        with open(json_path, 'r') as jf:
            label_content = json.load(jf)

        self.filter_lanes_by_size(label_content, min_height=min_lane_height)
        self.filter_few_markers(label_content, min_markers=2)
        self.fix_lane_names(label_content)

        content = {
            'projection_matrix': label_content['projection_matrix'],
            'lanes': label_content['lanes']
        }

        for lane in content['lanes']:
            for marker in lane['markers']:
                for pixel_key in marker['pixel_start'].keys():
                    marker['pixel_start'][pixel_key] = int(marker['pixel_start'][pixel_key])
                for pixel_key in marker['pixel_end'].keys():
                    marker['pixel_end'][pixel_key] = int(marker['pixel_end'][pixel_key])
                for pixel_key in marker['world_start'].keys():
                    marker['world_start'][pixel_key] = float(marker['world_start'][pixel_key])
                for pixel_key in marker['world_end'].keys():
                    marker['world_end'][pixel_key] = float(marker['world_end'][pixel_key])
        return content

    def get_files_from_folder(self, directory, extension=None):
        """Get all files within a folder that fit the extension """
        # NOTE Can be replaced by glob for newer python versions
        label_files = []
        for root, _, files in os.walk(directory):
            for some_file in files:
                label_files.append(os.path.abspath(os.path.join(root, some_file)))
        if extension is not None:
            label_files = list(filter(lambda x: x.endswith(extension), label_files))
        return label_files

    def get_base_name(self, input_path):
        """ /foo/bar/test/folder/image_label.ext --> test/folder/image_label.ext """
        return '/'.join(input_path.split('/')[-2:])

    def read_image(self, json_path):
        """ Reads image corresponding to json file

        Parameters
        ----------
        json_path: str
                path to json file / label
        image_type: str
                    type of image to read, either 'gray' or 'color'

        Returns
        -------
        numpy.array
            Image corresponding to image file

        Raises
        ------
        ValueError
            If image_type is neither 'gray' nor 'color'
        IOError
            If image_path does not exist. The image folder may not exist
            or may not be set in dataset_constants.py
        """
        # NOTE The function is built like this because extensions offer access to other types
        base_name = self.get_base_name(json_path)
        image_path = os.path.join(self.img_path, base_name.replace('.json', '_color_rect.png'))
        imread_code = cv2.IMREAD_COLOR

        if not os.path.exists(image_path):
            raise IOError(
                'Image does not exist: {}\n. Did you set dataset_constants.py?'.format(image_path))
        return cv2.imread(image_path, imread_code)

    def project_point(self, point, projection_matrix):
        """Projects 3D point into image coordinates

        Parameters
        ----------
        p1: iterable
            (x, y, z), line start in 3D
        p2: iterable
            (x, y, z), line end in 3D
        width: float
            width of marker in cm
        projection matrix: numpy.array, shape=(3, 3)
                        projection 3D location into image space

        Returns (x, y)
        """
        point = np.asarray(point)
        projection_matrix = np.asarray(projection_matrix)

        point_projected = projection_matrix.dot(point)
        point_projected /= point_projected[2]

        return point_projected

    def project_lane_marker(self, p1, p2, width, projection_matrix, color, img):
        """ Draws a marker by two 3D points (p1, p2) in 2D image space

        p1 and p2 are projected into the image space using a given projection_matrix.
        The line is given a fixed width (in cm) to be drawn. Since the marker width
        is given for the 3D space, the closer edge will be thicker in the image.
        The color can be freely set, e.g. according to lane association.

        Parameters
        ----------
        p1: iterable
            (x, y, z), line start in 3D
        p2: iterable
            (x, y, z), line end in 3D
        width: float
            width of marker in m, default=0.1 m
        projection matrix: numpy.array, shape=(3, 3)
                        projection 3D location into image space
        color: int or tuple
            color of marker, e.g. 255 or (0, 255, 255) in (b, g, r)
        img: numpy.array (dtype=numpy.uint8)
            Image array to draw the marker

        Notes
        ------
        You can't draw colored lines into a grayscale image.
        """
        p1 = np.asarray(p1)
        p2 = np.asarray(p2)

        p1_projected = self.project_point(p1, projection_matrix)
        p2_projected = self.project_point(p2, projection_matrix)

        points = np.zeros((4, 2), dtype=np.float32)
        shift = 0
        # shift_multiplier = static_cast<double>(1 << shift)
        shift_multiplier = 1  # simplified

        projection_matrix = np.asarray(projection_matrix)
        projected_half_width1 = projection_matrix[0, 0] * width / p1[2] / 2.0
        points[0, 0] = (p1_projected[0] - projected_half_width1) * shift_multiplier
        points[0, 1] = p1_projected[1] * shift_multiplier
        points[1, 0] = (p1_projected[0] + projected_half_width1) * shift_multiplier
        points[1, 1] = p1_projected[1] * shift_multiplier

        projected_half_width2 = projection_matrix[0, 0] * width / p2[2] / 2.0
        points[2, 0] = (p2_projected[0] + projected_half_width2) * shift_multiplier
        points[2, 1] = p2_projected[1] * shift_multiplier
        points[3, 0] = (p2_projected[0] - projected_half_width2) * shift_multiplier
        points[3, 1] = p2_projected[1] * shift_multiplier

        points = np.round(points).astype(np.int32)

        if not points[0, 1] == points[3, 1]:
            try:  # difference in cv2 versions
                aliasing = cv2.LINE_AA
            except AttributeError:
                aliasing = cv2.CV_AA
            cv2.fillConvexPoly(img, points, color, aliasing, shift)
            cv2.fillConvexPoly(img, points, color, aliasing, shift)

    def create_segmentation_image(self, json_path, color=None, image=None):
        """ Draws pixel-level markers onto image

        Parameters
        ----------
        json_path: str
                path to label-file
        color: int/uint8 for grayscale color to draw markers
            tuple (uint8, uint8, uint8), BGR values
            None for default marker colors, multi-class
        image: str, 'blank' for all zeros or 'gray' for gray image
            numpy.array, direct image input

        Returns:
        --------
        numpy.array
            image with drawn markers

        Notes
        -----
        This one is for visualizing the label, may not be optimal for training label creation
        """

        label = self.read_json(json_path)

        # TODO replace section by label_file_scripts read_image
        # NOTE Same in function above
        if isinstance(image, str):
            if image == 'blank':
                image = np.zeros((717, 1276), dtype=np.uint8)
            elif image == 'gray':
                image = self.read_image(json_path)
            # TODO Add color
            elif image == 'color':
                image = self.read_image(json_path)
            else:
                raise ValueError('Unknown image type {}'.format(image))

        if (len(image.shape) == 2 or image.shape[2] == 1)\
                and (color is None or not isinstance(color, int)):
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        for lane in label['lanes']:
            lane_id = lane['lane_id']
            for marker in lane['markers']:
                p1 = marker['world_start']
                p1 = [p1['x'], p1['y'], p1['z']]
                p2 = marker['world_end']
                p2 = [p2['x'], p2['y'], p2['z']]
                dcolor = DICT_COLORS[lane_id] if color is None else color
                self.project_lane_marker(
                    p1, p2, width=.1, projection_matrix=label['projection_matrix'],
                    color=dcolor, img=image)
        return image

    def __getitem__(self, idx):
        img = self.read_image(self.label_paths[idx])
        if self.b_vis:
            style = 'color'
        else:
            style = 'blank'

        seg_img_label = self.create_segmentation_image(
                    self.label_paths[idx], image=style)
        if self.b_vis:
            cv2.imwrite(os.path.join("/hdd/xfwang_data/vis", str(idx) + '.png'), seg_img_label)
        seg_img_label = cv2.cvtColor(seg_img_label, cv2.COLOR_BGR2GRAY)
        seg_img_label = seg_img_label > 0
        seg_img_label = seg_img_label.astype(np.uint8)
        seg_img_label = np.array(seg_img_label)
        anno = seg_img_label

        data = {
            "img": img,
            "anno": anno,
        }

        if self.transforms is not None:
            for transform in self.transforms:
                data = transform(data)
        return data
        
    def __len__(self):
        return len(self.label_paths)


if __name__ == '__main__':
    data_rootdir = "/hdd/xfwang_data/waic-bosch"
    input_hw = resize_hw = (640, 1024) 
    inter_method = 10
    pixel_center_aligned = False
    min_valid_clip_area_ratio = 0.5
    rand_translation_ratio = 0.1
    dataset = WaicBoxyDataset(img_path=os.path.join(data_rootdir, "WAIC-bosch-val/boxy"), 
                              label_file=os.path.join(data_rootdir, "WAIC-bosch-val/boxy_labels_valid.json"),
                              transforms=[
                                dict(
                                    type="IterableDetRoITransform",
                                    # roi transform
                                    target_wh=input_hw[::-1],
                                    resize_wh=None
                                    if resize_hw is None
                                    else resize_hw[::-1],
                                    img_scale_range=(0.7, 1.0 / 0.7),
                                    roi_scale_range=(0.5, 2.0),
                                    min_sample_num=1,
                                    max_sample_num=1,
                                    center_aligned=False,
                                    inter_method=inter_method,
                                    use_pyramid=True,
                                    pyramid_min_step=0.7,
                                    pyramid_max_step=0.8,
                                    pixel_center_aligned=pixel_center_aligned,
                                    min_valid_area=100,
                                    min_valid_clip_area_ratio=min_valid_clip_area_ratio,
                                    min_edge_size=10,
                                    rand_translation_ratio=rand_translation_ratio,
                                    rand_aspect_ratio=0.0,
                                    rand_rotation_angle=0,
                                    flip_prob=0.5,
                                    clip_bbox=True,
                                    keep_aspect_ratio=True,
                                ),
                                dict(
                                    type="PadDetData",
                                    max_gt_boxes_num=200,
                                    max_ig_regions_num=100,
                                ),
                                ],
                              b_vis=True)

    # data_rootdir = "/hdd/xfwang_data/waic-bosch"
    # dataset = WaicLaneDataset(img_path=os.path.join(data_rootdir, "WAIC-bosch-train/llamas"), 
    #                           input_folder=os.path.join(data_rootdir, "WAIC-bosch-train/llamas-label"),
    #                           b_vis=True)

    for data in dataset:
        print('yes') 