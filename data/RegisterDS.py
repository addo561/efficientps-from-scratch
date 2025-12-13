from detectron2.data.datasets import register_coco_instances
trainR = register_coco_instances("YourTrainDatasetName", {},"path to train.json", "path to train image folder")
testR = register_coco_instances("YourTestDatasetName", {}, "path to test.json", "path to test image folder")

