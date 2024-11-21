from simanneal import Annealer
import math

import sqlite3
import random
import argparse
import json
import os
import sys

sys.path.append('/media/asghar/media/FutureDet-NAS')
sys.path.append('/media/asghar/media/FutureDet-NAS/Core/nuscenes-forecast/python-sdk')
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaWarning)

import numpy as np
import torch
import yaml
from det3d.datasets import build_dataset
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.apis import (
    build_optimizer,
    get_root_logger,
    init_dist,
    set_random_seed,
    train_detector,
)
import pdb


#  --work_dir
# --seed 0
# configs/centerpoint/nusc_centerpoint_forecast_n0_detection.py
# '--work_dir models/FutureDetection/nusc_centerpoint_forecast_n0_detection'
'python  ./tools/train.py configs/centerpoint/nusc_centerpoint_forecast_n0_detection.py --seed 0 --work_dir models/FutureDetection/nusc_centerpoint_forecast_n0_detection'
def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("--config", help="train config file path",default='/media/asghar/media/FutureDet-NAS/configs/centerpoint/nusc_centerpoint_forecast_n3dtf_detection.py')
    parser.add_argument("--work_dir", help="the dir to save logs and models")
    # parser.add_argument("config", default='../configs/centerpoint/nusc_centerpoint_forecast_n0_detection.py',help="train config file path")
    #parser.add_argument("--work_dir", default='models/FutureDetection/nusc_centerpoint_forecast_n0_detection', help="the dir to save logs and model")
    parser.add_argument("--resume_from", help="the checkpoint file to resume from")
    parser.add_argument("--validate", action="store_true", help="whether to evaluate the checkpoint during training")
    parser.add_argument("--gpus", type=int, default=1,
                        help="number of gpus to use " "(only applicable to non-distributed training)", )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm", "mpi"], default="none",
                        help="job launcher", )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--autoscale-lr", action="store_true",
                        help="automatically scale lr with the number of gpus", )
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


class SimAnealler(Annealer):
    def __init__(self,state,dir, init_model):
        super(SimAnealler,self).__init__(state)
        self.num = 0
        self.last_e=math.inf
        self.past_model=init_model
        self.best=math.inf
        self.path=dir

        self.db=dir+'/models.db'
        print(self.db)

        conn=sqlite3.connect(self.db)

        c=conn.cursor()

        c.execute('''CREATE TABLE bests (num int, conf test, energy real)''')
        conn.commit()
        c.execute('''CREATE TABLE _all_ (num int, conf test, energy real)''')
        conn.commit()
        conn.close()

    def move(self):
        if True:
            if random.choice([True,False]):

                self.state['neck']['layer_nums']=[self.state['neck']['layer_nums'][0]+random.choice([-1,1]),self.state['neck']['layer_nums'][1]]
            else:
                self.state['neck']['layer_nums'] = [self.state['neck']['layer_nums'][0],self.state['neck']['layer_nums'][1]+random.choice([-1,1])]

        else:
            change=random.choice(['reg','height','dim','rot','vel'])
            if random.choice([True,False]):

                self.state['bbox_head']['common_heads'][change]=[self.state['bbox_head']['common_heads'][change][0]+random.choice([-1,1]),self.state['bbox_head']['common_heads'][change][1]]
            else:
                self.state['bbox_head']['common_heads'][change] = [self.state['bbox_head']['common_heads'][change][0],self.state['bbox_head']['common_heads'][change][1]+random.choice([-1,1])]


        return self.energy()

    def energy(self):
        print(self.state,'ewggrrhgerwh')
        logger = get_root_logger('info')
        logger.info("Distributed training: {}".format(distributed))
        logger.info(f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")
        e=math.inf
        model = build_detector(self.state,train_cfg=cfg.train_cfg,test_cfg=cfg.test_cfg,structure=None)
        print('model',model)
        datasets = [build_dataset(cfg.data.train)]

        if len(cfg.workflow) == 2:
            datasets.append(build_dataset(cfg.data.val))

        if cfg.checkpoint_config is not None:
            # save det3d version, config file content and class names in
            # checkpoints as meta data
            cfg.checkpoint_config.meta = dict(
                config=cfg.text, CLASSES=datasets[0].CLASSES
            )

        # add an attribute for visualization convenience
        model.CLASSES = datasets[0].CLASSES
        latecny = train_detector(
            model,
            datasets,
            cfg,
            distributed=distributed,
            validate=args.validate,
            logger=logger,
        )
        print(model)
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params=sum([np.prod(p.size()) for p in model_parameters])
        print(params)
        print(latecny)
        e=float(latecny)
        print(e)
        print(type(e))
        conn = sqlite3.connect(self.db)
        c = conn.cursor()
        c.execute('''INSERT INTO _all_ VALUES (?,?,?)''',
                  [self.num, str(self.state), e])

        conn.commit()
        conn.close()
        if e < self.best:
            conn = sqlite3.connect(self.db)
            c = conn.cursor()
            c.execute('''INSERT INTO bests VALUES (?,?,?)''',
                  [self.num, str(self.state), e])
            conn.commit()
            conn.close()
            self.best = e
        self.num = self.num + 1
        #sys.exit()

        return e


def Sim_Annealer(init,init_model):
    path ='/media/asghar/models/test101'
    print('fef')
    print(init)
    tsp = SimAnealler(init,path,init_model)
    tsp.Tmax = 25000.0
    tsp.Tmin = 25.0
    tsp.copy_strategy = "deepcopy"
    state, e = tsp.anneal()
    return state

if __name__=='__main__':
    args = parse_args()
    print("RANK: {}".format(args.local_rank))
    cfg = Config.fromfile(args.config)
    cfg.local_rank = args.local_rank
    torch.cuda.set_device(args.local_rank)
    distributed = False
    if "WORLD_SIZE" in os.environ:
        distributed = int(os.environ["WORLD_SIZE"]) > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

        cfg.gpus = torch.distributed.get_world_size()

    if args.autoscale_lr:
        cfg.lr_config.lr_max = cfg.lr_config.lr_max * cfg.gpus

    logger = get_root_logger(cfg.log_level)
    logger.info("Distributed training: {}".format(distributed))
    logger.info(f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")

    if args.local_rank == 0:
        # copy important files to backup
        backup_dir = os.path.join(cfg.work_dir, "det3d")
        os.makedirs(backup_dir, exist_ok=True)
        # os.system("cp -r * %s/" % backup_dir)
        # logger.info(f"Backup source files to {cfg.work_dir}/det3d")

    if args.seed is not None:
        logger.info("Set random seed to {}".format(args.seed))
        set_random_seed(args.seed)

    #print(cfg.model)
    #sys.exit()
    structure=[]

    #print(type(cfg.model))
    #sys.exit()
    #print(cfg.model['neck']['layer_nums'])
    #print(cfg.model['bbox_head']['common_heads'])
    #sys.exit()
    SA_State=Sim_Annealer(cfg.model,None)