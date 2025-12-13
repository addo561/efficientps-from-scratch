###MAIN MODEL

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.structures import ImageList
from detectron2.modeling import META_ARCH_REGISTRY, GeneralizedRCNN, build_sem_seg_head
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.utils.memory import retry_if_cuda_oom

@META_ARCH_REGISTRY.register()
class EfficientPS(GeneralizedRCNN):
    def __init__(self,cfg):
        super().__init__(cfg)
        #Build  Semantic Head
        self.seg_head = build_sem_seg_head(cfg,self.backbone.output_shape())
        #fusion params
        self.panoptic_fusion_overlap_thresh = 0.5
        self.to(self.device)

    def forward(self,input):
        if not self.training:
            return self.inference(input)

        images = self.preprocess_image(input)
        #  Ground Truth
        if "instances" in input[0]:
            gt_instances = [x["instances"].to(self.device) for x in input]
        else:
            gt_instances = None
            
        if "sem_seg" in input[0]:
            gt_sem_seg = [x["sem_seg"].to(self.device) for x in input]
            # Stack and add channel dim: [Batch, 1, H, W]
            gt_sem_seg = torch.stack(gt_sem_seg).unsqueeze(1)
        else:
            gt_sem_seg = None

        #extract features
        features = self.backbone(images.tensor)  

        #run semantic head
        sem_seg_losses = self.sem_seg_head(features, gt_sem_seg) 

        #Run Instance Branch (RPN + ROI Heads)
        # GeneralizedRCNN separates proposal generation and ROI processing
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)

        # 5. Combine All Losses
        losses = {}
        losses.update(sem_seg_losses)
        losses.update(proposal_losses)
        losses.update(detector_losses)

        return  losses

    def inference(self, batched_inputs):
        """
        Inference Logic: Predict -> Fuse -> Return
        """
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        
        # Semantic Inference (Returns raw logits: [B, C, H, W])
        sem_seg_results = self.sem_seg_head(features, None)
        
        # Instance Inference
        proposals, _ = self.proposal_generator(images, features, None)
        instance_results, _ = self.roi_heads(images, features, proposals, None)
        
        #Post-processing and Fusion
        results = []
        for i, (sem_res, inst_res) in enumerate(zip(sem_seg_results, instance_results)):
            
            # A. Resize semantic output to original image size
            h, w = batched_inputs[i]["height"], batched_inputs[i]["width"]
            image_size = (h, w)
            sem_seg_res = sem_seg_postprocess(sem_res, image_size, h, w)
            
            # B. Combine (Fuse) Semantic + Instance
            panoptic_r = self.combine_semantic_and_instance(sem_seg_res, inst_res)
            
            results.append({
                "panoptic_seg": panoptic_r,
                "instances": inst_res,
                "sem_seg": sem_seg_res
            })
            
        return results

    def combine_semantic_and_instance(self, sem_seg, instances):
        """
        Simple Panoptic Fusion Strategy (Heuristic):
        1. The Semantic Segmentation map (argmaxed).
        2. Overlay Instance Masks on top (sorted by confidence).
        """
        # 1. Get Semantic Prediction (Shape: [H, W])
        # sem_seg is [C, H, W] logits, we need the class indices
        sem_seg = torch.argmax(sem_seg, dim=0)
        
        panoptic_seg = torch.zeros_like(sem_seg, dtype=torch.int32)
        segments_info = []
        
        current_segment_id = 1
        
        # 2. Process Instances (High confidence first)
        instances = instances[instances.scores > 0.5] 
        
        # Paste instances
        if len(instances) > 0:
            masks = instances.pred_masks
            classes = instances.pred_classes
            scores = instances.scores
            
            # Sort by score (descending) so best masks are pasted last/first logic depending on overlap
            
            for j in range(len(instances)):
                mask = masks[j]
                class_id = classes[j]
                
                # Check for mask validity
                if mask.sum() == 0: continue
                
                # Assign a unique ID for the panoptic map
                panoptic_seg[mask] = current_segment_id
                
                segments_info.append({
                    "id": current_segment_id,
                    "isthing": True,
                    "category_id": class_id.item(),
                    "score": scores[j].item()
                })
                current_segment_id += 1
                
        # 3. Process Stuff (Background/Semantic)
        remaining_mask = (panoptic_seg == 0)
        
        
        
        semantic_labels = sem_seg[remaining_mask].unique()
        for label in semantic_labels:
            label = label.item()
            if label == 0: continue 
            
            mask = (sem_seg == label) & remaining_mask
            if mask.sum() > 0:
                panoptic_seg[mask] = current_segment_id
                segments_info.append({
                    "id": current_segment_id,
                    "isthing": False,
                    "category_id": label
                })
                current_segment_id += 1
                
        return (panoptic_seg, segments_info)   