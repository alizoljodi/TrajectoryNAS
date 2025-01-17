import math
import random
import sqlite3
import argparse
import os
import sys
import warnings

from simanneal import Annealer
from numba.core.errors import NumbaDeprecationWarning, NumbaWarning
import numpy as np
import torch
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

# Suppress warnings
warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaWarning)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a detector using Simulated Annealing"
    )
    parser.add_argument(
        "--config",
        help="Train config file path",
        default="/media/asghar/media/FutureDet-NAS/configs/centerpoint/nusc_centerpoint_forecast_n3dtf_detection.py",
    )
    parser.add_argument("--work_dir", help="Directory to save logs and models")
    parser.add_argument("--resume_from", help="Checkpoint file to resume from")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Evaluate the checkpoint during training",
    )
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="Job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--autoscale-lr",
        action="store_true",
        help="Automatically scale learning rate with GPUs",
    )
    args = parser.parse_args()

    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


class SimulatedAnnealer(Annealer):
    def __init__(self, state, directory, init_model):
        super().__init__(state)
        self.num = 0
        self.best_energy = math.inf
        self.init_model = init_model
        self.directory = directory
        self.db_path = os.path.join(directory, "models.db")

        # Initialize SQLite database
        self._initialize_database()

    def _initialize_database(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            """CREATE TABLE IF NOT EXISTS bests (iteration INT, config TEXT, energy REAL)"""
        )
        c.execute(
            """CREATE TABLE IF NOT EXISTS all_trials (iteration INT, config TEXT, energy REAL)"""
        )
        conn.commit()
        conn.close()

    def move(self):
        if random.choice([True, False]):
            layer_idx = random.choice([0, 1])
            self.state["neck"]["layer_nums"][layer_idx] += random.choice([-1, 1])
        else:
            head_key = random.choice(["reg", "height", "dim", "rot", "vel"])
            dimension_idx = random.choice([0, 1])
            self.state["bbox_head"]["common_heads"][head_key][
                dimension_idx
            ] += random.choice([-1, 1])

        return self.energy()

    def energy(self):
        logger = get_root_logger("info")
        logger.info(f"Training model with state: {self.state}")
        model = build_detector(
            self.state, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg
        )

        datasets = [build_dataset(cfg.data.train)]
        if len(cfg.workflow) == 2:
            datasets.append(build_dataset(cfg.data.val))

        # Train the model
        latency = train_detector(
            model,
            datasets,
            cfg,
            distributed=distributed,
            validate=args.validate,
            logger=logger,
        )

        energy_value = float(latency)
        self._log_trial(energy_value)

        if energy_value < self.best_energy:
            self.best_energy = energy_value
            self._log_best(energy_value)

        self.num += 1
        return energy_value

    def _log_trial(self, energy):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            "INSERT INTO all_trials VALUES (?, ?, ?)",
            (self.num, str(self.state), energy),
        )
        conn.commit()
        conn.close()

    def _log_best(self, energy):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            "INSERT INTO bests VALUES (?, ?, ?)", (self.num, str(self.state), energy)
        )
        conn.commit()
        conn.close()


def run_simulated_annealing(init_state, init_model):
    output_path = "/media/asghar/models/test101"
    annealer = SimulatedAnnealer(init_state, output_path, init_model)
    annealer.Tmax = 25000.0
    annealer.Tmin = 25.0
    annealer.steps = 1000
    annealer.copy_strategy = "deepcopy"
    best_state, best_energy = annealer.anneal()
    return best_state, best_energy


if __name__ == "__main__":
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.local_rank = args.local_rank
    torch.cuda.set_device(args.local_rank)

    distributed = os.environ.get("WORLD_SIZE", "0") > "1"
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        cfg.gpus = torch.distributed.get_world_size()

    if args.autoscale_lr:
        cfg.lr_config.lr_max *= cfg.gpus

    if args.seed is not None:
        set_random_seed(args.seed)

    initial_state = cfg.model
    run_simulated_annealing(initial_state, None)
