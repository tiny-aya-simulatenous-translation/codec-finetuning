"""Unit tests for core training infrastructure: schedulers, optimizers, and augmentation.

These tests verify behaviour contracts that the training loop depends on,
without requiring a GPU or real datasets.  They run as part of the CI
suite via ``uv run pytest tests/``.

Test classes
------------
- :class:`SchedulerBehaviorTests` -- Learning-rate scheduler warmup,
  cosine decay, and checkpoint round-tripping.
- :class:`OptimizerIntegrationTests` -- :class:`~train.optimizer_factory.MuonHybridOptimizer`
  param-group sharing with inner optimizers.
- :class:`AugmentationReproducibilityTests` -- Deterministic augmentation
  output when the global random seed is fixed.
"""

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
    """Verify learning-rate scheduler correctness and persistence."""

    def test_cosine_warmup_transitions_to_decay(self) -> None:
        """LR should increase during warmup, then decay after peak."""
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
        # Collect LR at each step: expect monotonic rise then decline.
        lrs = [optimizer.param_groups[0]["lr"]]
        for _ in range(6):
            optimizer.step()
            scheduler.step()
            lrs.append(optimizer.param_groups[0]["lr"])

        # Warmup phase: LR increases on each of the first 3 steps.
        self.assertLess(lrs[0], lrs[1])
        self.assertLess(lrs[1], lrs[2])
        # Peak at step 3 should equal the base LR.
        self.assertAlmostEqual(lrs[2], 1.0, places=6)
        # Cosine decay: final LR should be lower than peak.
        self.assertLess(lrs[-1], lrs[2])

    def test_checkpoints_persist_scheduler_state(self) -> None:
        """Saved checkpoints must include ``scheduler_state_dict``."""
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
    """Verify MuonHybridOptimizer shares param groups correctly."""

    def test_muon_hybrid_shares_param_groups_with_inner_optimizers(self) -> None:
        """LR changes on the hybrid must propagate to inner optimizers."""
        first = torch.nn.Parameter(torch.tensor(1.0))
        second = torch.nn.Parameter(torch.tensor(2.0))
        muon_like = torch.optim.SGD([first], lr=1.0)
        adamw = torch.optim.AdamW([second], lr=0.5)

        optimizer = MuonHybridOptimizer(muon_like, adamw)
        # Applying a LambdaLR to the hybrid should scale the inner LRs.
        LambdaLR(optimizer, lr_lambda=lambda step: 0.5)

        self.assertAlmostEqual(muon_like.param_groups[0]["lr"], 0.5)
        self.assertAlmostEqual(adamw.param_groups[0]["lr"], 0.25)

        # Direct mutation of param_groups must be visible to the inner
        # optimizer because the lists are shared (not copied).
        optimizer.param_groups[0]["lr"] = 0.2
        self.assertAlmostEqual(muon_like.param_groups[0]["lr"], 0.2)


class AugmentationReproducibilityTests(unittest.TestCase):
    """Verify that augmentation is deterministic given the same seed."""

    def test_default_rng_respects_global_random_seed(self) -> None:
        """Two calls with the same global seed must produce identical output."""
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
