import sys

from det3d.utils import build_from_cfg
from torch import nn

from .registry import (
    BACKBONES,
    DETECTORS,
    HEADS,
    LOSSES,
    NECKS,
    READERS,
    SECOND_STAGE,
    ROI_HEAD
)


def build(cfg, registry, default_args=None,structure=None):

    if isinstance(cfg, list):
        modules = [build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg]
        return nn.Sequential(*modules)
    else:
        print(registry)
        #sys.exit()
        return build_from_cfg(cfg, registry, default_args,structure=structure)

def build_second_stage_module(cfg):
    return build(cfg, SECOND_STAGE)

def build_roi_head(cfg):
    return build(cfg, ROI_HEAD)


def build_reader(cfg):
    return build(cfg, READERS)


def build_backbone(cfg):
    return build(cfg, BACKBONES)


def build_neck(cfg):
    return build(cfg, NECKS)

def build_head(cfg):
    return build(cfg, HEADS)


def build_loss(cfg):
    return build(cfg, LOSSES)


def build_detector(cfg, train_cfg=None, test_cfg=None,structure=None):
    print(structure)
    #sys.exit()
    return build(cfg, DETECTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg),structure=structure)
