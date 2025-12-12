
from detectron2.config import get_cfg
from detectron2 import model_zoo

_C = get_cfg()
_C.MODEL.SEM_SEG_HEAD.CLASS_WEIGHTS = []

def  setup_efficient_config(num_classes,weights):
    cfg =  get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
    cfg.MODEL.ROI_HEADS.IN_FEATURES = ["P2", "P3", "P4", "P5"]
    cfg.MODEL.RPN.IN_FEATURES = ["P2", "P3", "P4", "P5"]
    cfg.MODEL.PROPOSAL_GENERATOR.NAME = "RPN"
    # EfficientPS typically uses standard anchor sizes/ratios
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32], [64], [128], [256], [512]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]

    # ----------------------------------------------------------------
    #  ROI HEADS (Box, Class, Mask)
    # ----------------------------------------------------------------
    cfg.MODEL.ROI_HEADS.NAME = "StandardROIHeads"
    
    # BOX HEAD (Class + Box): 
    # Paper uses standard 2 FC layers.
    cfg.MODEL.ROI_BOX_HEAD.NAME = "FastRCNNConvFCHead"
    cfg.MODEL.ROI_BOX_HEAD.NUM_FC = 2
    cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 7
    
    # MASK HEAD: 
    # custom one
    cfg.MODEL.ROI_MASK_HEAD.NAME = "EfficientPSMaskHead"
    cfg.MODEL.ROI_MASK_HEAD.NUM_CONV = 4
    cfg.MODEL.ROI_MASK_HEAD.CONV_DIM = 256
    cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 14


    # ----------------------------------------------------------------
    #  Semantic Head.
    # ----------------------------------------------------------------
    cfg.MODEL.SEM_SEG_HEAD.NAME = 'SemanticHead'
    cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES = ["P2", "P3", "P4", "P5"]
    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE = 255
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_classes
    cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM = 128
    cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE = 4
    cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT = 1.0

    cfg. MODEL.SEM_SEG_HEAD.CLASS_WEIGHTS = weights

    return cfg