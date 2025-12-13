from detectron2.data.datasets import register_coco_instances
from  detectron2.data import MetadataCatalog,DatasetCatalog
import  random
import cv2
import os
from pathlib import Path
from detectron2.utils.visualizer import Visualizer

for name in ["trainDS", "testDs"]:
    if name in DatasetCatalog.list():
        DatasetCatalog.remove(name)
    if name in MetadataCatalog.list():
        MetadataCatalog.remove(name)


base = Path(__file__).resolve().parent
trainR = register_coco_instances("trainDS", {},f"{base}/Panoptic-Segmentation/train/_annotations.coco.json", f"{base}/Panoptic-Segmentation/train")
testR = register_coco_instances("testDs", {}, f"{base}/Panoptic-Segmentation/test/_annotations.coco.json", f"{base}/Panoptic-Segmentation/test")
train_metadata = MetadataCatalog.get("trainDS")
data_dicts = DatasetCatalog.get("trainDS")
for d in random.sample(data_dicts,4 ):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=train_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    cv2.imshow('Image',out.get_image()[:, :, ::-1])
    cv2.waitKey(0)