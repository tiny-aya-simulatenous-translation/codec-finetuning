import random
import tempfile
import unittest
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LambdaLR

from train.optimizer_factory import MuonHybridOptimizer
from train.train_mimi import _create_scheduler, save_checkpoint
from train.utils.augmentation import AugmentationConfig, augment_waveform


class SchedulerBehaviorTests(unittest.TestCase):
    def test_cosine_warmup_transitions_to_decay(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=1.0)
        config = {
            "training": {"max_steps": 10},
            "scheduler": {
                "name": "cosine",
                "warmup_steps": 3,
                "min_lr_ratio": 0.1,
            },
        }

        scheduler = _create_scheduler(optimizer, config, metadata={})

        self.assertIsNotNone(scheduler)
        lrs = [optimizer.param_groups[0]["lr"]]
        for _ in range(6):
            optimizer.step()
            scheduler.step()
            lrs.append(optimizer.param_groups[0]["lr"])

        self.assertLess(lrs[0], lrs[1])
        self.assertLess(lrs[1], lrs[2])
        self.assertAlmostEqual(lrs[2], 1.0, places=6)
        self.assertLess(lrs[-1], lrs[2])

    def test_checkpoints_persist_scheduler_state(self) -> None:
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 0.5 ** step)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"output_dir": tmpdir}
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                ema=None,
                step=3,
                config=config,
            )

            checkpoint = torch.load(
                Path(tmpdir) / "checkpoint_step_3.pt",
                map_location="cpu",
            )

        self.assertIn("scheduler_state_dict", checkpoint)


class OptimizerIntegrationTests(unittest.TestCase):
    def test_muon_hybrid_shares_param_groups_with_inner_optimizers(self) -> None:
        first = torch.nn.Parameter(torch.tensor(1.0))
        second = torch.nn.Parameter(torch.tensor(2.0))
        muon_like = torch.optim.SGD([first], lr=1.0)
        adamw = torch.optim.AdamW([second], lr=0.5)

        optimizer = MuonHybridOptimizer(muon_like, adamw)
        LambdaLR(optimizer, lr_lambda=lambda step: 0.5)

        self.assertAlmostEqual(muon_like.param_groups[0]["lr"], 0.5)
        self.assertAlmostEqual(adamw.param_groups[0]["lr"], 0.25)

        optimizer.param_groups[0]["lr"] = 0.2
        self.assertAlmostEqual(muon_like.param_groups[0]["lr"], 0.2)


class AugmentationReproducibilityTests(unittest.TestCase):
    def test_default_rng_respects_global_random_seed(self) -> None:
        waveform = torch.ones(1, 32)
        config = AugmentationConfig(
            preset="custom",
            gain_db=(0.0, 1.0),
        )

        random.seed(123)
        first = augment_waveform(waveform.clone(), 24_000, config)

        random.seed(123)
        second = augment_waveform(waveform.clone(), 24_000, config)

        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
