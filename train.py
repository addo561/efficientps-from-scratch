# train_net.py
import os
from detectron2.engine import DefaultTrainer, default_argument_parser, launch, default_setup
from detectron2.checkpoint import DetectionCheckpointer

# This triggers the @REGISTRY.register() 
from models.efficientps import EfficientPS
from models.neck.bifpn import biFPN
from models.heads.instance_head import EfficientPSMaskHead
from models.heads.semantic_head import SemanticHead
from configs.config import setup_efficient_config

def main(args):
    # 1. Setup Config
    # We pass the class weights here if you have them calculated, else []
    cfg = setup_efficient_config(num_classes=80) 
    
    # Override defaults with command line args if needed (e.g. changing learning rate)
    cfg.merge_from_list(args.opts)
    
    # Setup logger and environment
    default_setup(cfg, args)
    
    # 2. Create Trainer
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    
    # 3. Train
    return trainer.train()

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )