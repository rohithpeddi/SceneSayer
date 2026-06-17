import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class _AGConstants:
    BACKGROUND = "__background__"
    ANNOTATIONS = "annotations"
    FRAMES = "frames"
    OBJECT_CLASSES_FILE = "object_classes.txt"
    PERSON_BOUNDING_BOX_PKL = "person_bbox.pkl"
    OBJECT_BOUNDING_BOX_RELATIONSHIP_PKL = "object_bbox_and_relationship.pkl"
    METADATA = "metadata"
    SET = "set"
    BOUNDING_BOX = "bbox"
    CLASS = "class"
    VISIBLE = "visible"


class ActionGenomeSceneSayerDetectorDataset(Dataset):
    """
    Action Genome dataset for training a torchvision Faster R-CNN detector in a way
    that is easier to bridge back into SceneSayer later.

    Key choices:
    - uses original image resolution
    - returns raw RGB tensors in [0, 1]
    - keeps GT boxes in original image coordinates
    - lets the detector's own transform perform the resize policy
    """

    def __init__(
            self,
            data_path: str,
            phase: str = "train",
            datasize: str = "full",
            filter_nonperson_box_frame: bool = True,
            filter_small_box: bool = False,
    ):
        self.data_path = data_path
        self.phase = phase
        self.datasize = datasize
        self.filter_nonperson_box_frame = filter_nonperson_box_frame
        self.filter_small_box = filter_small_box
        self.frames_path = os.path.join(self.data_path, _AGConstants.FRAMES)

        self.object_classes = self._fetch_object_classes()
        self.person_bbox, self.object_bbox = self._fetch_object_person_bboxes()
        self.samples = self._build_samples()

    def _fetch_object_classes(self) -> List[str]:
        object_classes = [_AGConstants.BACKGROUND]
        object_classes_path = os.path.join(
            self.data_path,
            _AGConstants.ANNOTATIONS,
            _AGConstants.OBJECT_CLASSES_FILE,
        )
        with open(object_classes_path, "r", encoding="utf-8") as handle:
            for line in handle:
                object_classes.append(line.strip("\n"))

        # Preserve SceneSayer's class-name normalization.
        object_classes[9] = "closet/cabinet"
        object_classes[11] = "cup/glass/bottle"
        object_classes[23] = "paper/notebook"
        object_classes[24] = "phone/camera"
        object_classes[31] = "sofa/couch"
        return object_classes

    def _fetch_object_person_bboxes(self) -> Tuple[Dict, Dict]:
        annotations_path = os.path.join(self.data_path, _AGConstants.ANNOTATIONS)
        with open(os.path.join(annotations_path, _AGConstants.PERSON_BOUNDING_BOX_PKL), "rb") as handle:
            person_bbox = pickle.load(handle)
        with open(os.path.join(annotations_path, _AGConstants.OBJECT_BOUNDING_BOX_RELATIONSHIP_PKL), "rb") as handle:
            object_bbox = pickle.load(handle)
        return person_bbox, object_bbox

    def _build_samples(self) -> List[Dict]:
        samples: List[Dict] = []
        frame_names = list(self.person_bbox.keys())
        if self.datasize == "mini":
            frame_names = frame_names[:80000]

        for frame_name in frame_names:
            objects = self.object_bbox[frame_name]
            if objects[0][_AGConstants.METADATA][_AGConstants.SET] != self.phase:
                continue

            person_boxes = np.array(
                self.person_bbox[frame_name][_AGConstants.BOUNDING_BOX],
                dtype=np.float32,
            ).reshape(-1, 4)
            if self.filter_nonperson_box_frame and len(person_boxes) == 0:
                continue

            frame_boxes: List[List[float]] = []
            frame_labels: List[int] = []

            for person_box in person_boxes:
                x1, y1, x2, y2 = person_box.tolist()
                if (x2 - x1) >= 1 and (y2 - y1) >= 1:
                    frame_boxes.append([x1, y1, x2, y2])
                    frame_labels.append(1)

            for obj in objects:
                if not obj[_AGConstants.VISIBLE] or obj[_AGConstants.BOUNDING_BOX] is None:
                    continue
                x, y, w, h = obj[_AGConstants.BOUNDING_BOX]
                x1, y1, x2, y2 = float(x), float(y), float(x + w), float(y + h)
                if (x2 - x1) < 1 or (y2 - y1) < 1:
                    continue
                frame_boxes.append([x1, y1, x2, y2])
                frame_labels.append(self.object_classes.index(obj[_AGConstants.CLASS]))

            if not frame_boxes:
                continue

            samples.append(
                {
                    "filename": frame_name,
                    "boxes": frame_boxes,
                    "labels": frame_labels,
                }
            )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image_path = os.path.join(self.frames_path, sample["filename"])
        image = Image.open(image_path).convert("RGB")
        image_np = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).contiguous()

        target = {
            "boxes": torch.tensor(sample["boxes"], dtype=torch.float32),
            "labels": torch.tensor(sample["labels"], dtype=torch.int64),
            "image_id": torch.tensor([index], dtype=torch.int64),
        }
        return image_tensor, target


def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets
