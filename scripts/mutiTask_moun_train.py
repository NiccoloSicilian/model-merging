import logging
import os
from typing import Dict, List, Union

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from nn_core.callbacks import NNTemplateCore
from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import seed_index_everything
from nn_core.model_logging import NNLogger

from model_merging.model.encoder import ImageEncoder
from model_merging.model.heads import get_classification_head
from model_merging.model.image_classifier import MultiTaskImageClassifier 
from model_merging.utils.io_utils import load_model_from_hf, upload_model_to_hf
from hydra.utils import instantiate

pylogger = logging.getLogger(__name__)
torch.set_float32_matmul_precision("high")

def run(cfg: DictConfig):
    seed_index_everything(cfg)

    template_core: NNTemplateCore = NNTemplateCore(
        restore_cfg=cfg.train.get("restore", None),
    )

    logger: NNLogger = NNLogger(
        logging_cfg=cfg.train.logging, cfg=cfg, resume_id=template_core.resume_id
    )

    zeroshot_encoder: ImageEncoder = load_model_from_hf(
        model_name=cfg.nn.encoder.model_name
    )
    
    # IMPORTANT: Unfreeze the encoder to train from scratch
    for param in zeroshot_encoder.parameters():
        param.requires_grad = True

    classification_heads = nn.ModuleDict()
    train_dataloaders = {}
    test_dataloaders = {}

    # Iterate over the LIST of datasets provided by the N14/N8 benchmark
    for task_config in cfg.benchmark.datasets:
        # We assume each dataset config has a 'name' attribute (e.g., 'SVHN', 'MNIST')
        task_name = task_config.name 
        
        # 1. Instantiate the specific classification head
        head = get_classification_head(
            cfg.nn.encoder.model_name,
            task_name,
            ckpt_path=cfg.misc.ckpt_path,
            openclip_cachedir=cfg.misc.openclip_cachedir,
            device=cfg.device,
        )
        
        # IMPORTANT: Unfreeze the head
        for param in head.parameters():
            param.requires_grad = True
            
        classification_heads[task_name] = head

        # 2. Instantiate the dataset
        task_dataset = instantiate(
            task_config, 
            preprocess_fn=zeroshot_encoder.val_preprocess,
            batch_size=cfg.train.batch_size,
        )
        train_dataloaders[task_name] = task_dataset.train_loader
        test_dataloaders[task_name] = task_dataset.test_loader

    # 3. Instantiate our custom MultiTask model
    model: MultiTaskImageClassifier = hydra.utils.instantiate(
        cfg.nn.module,
        encoder=zeroshot_encoder,
        classifiers=classification_heads, 
        _recursive_=False,
    )

    pylogger.info("Instantiating the <Trainer>")
    trainer = pl.Trainer(
        default_root_dir=cfg.core.storage_dir,
        logger=logger,
        enable_checkpointing=False,
        **cfg.train.trainer,
    )

    pylogger.info("Starting training!")
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloaders,
    )

    pylogger.info("Starting testing!")
    trainer.test(model=model, dataloaders=test_dataloaders)

    upload_model_to_hf(model.encoder, cfg.nn.encoder.model_name, "multitask_trained_from_scratch")

    logger.log_configuration(model, cfg)

    if logger is not None:
        logger.experiment.finish()
        
@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="finetune.yaml")
def main(cfg: omegaconf.DictConfig):
    run(cfg)

if __name__ == "__main__":
    main()
    
