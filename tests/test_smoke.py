import unittest
from pathlib import Path
import tempfile

import numpy as np
import torch

from src.eval.plot_distribution import plot_distribution_comparison
from src.data.dataset import create_paired_data_bundle
from src.data.transforms import BCTStandardizer
from src.losses.physics import species_bounds_hinge_loss
from src.losses.wgan_gp import gradient_penalty
from src.models.critic import Critic
from src.models.generator import Generator
from src.oracle.true_predictor import get_true_prediction


class TestTransforms(unittest.TestCase):
    def test_bct_inverse(self):
        x = np.abs(np.random.randn(128, 10).astype(np.float32)) + 1e-3
        tfm = BCTStandardizer(use_bct=True, standardize=True).fit(x)
        y = tfm.transform(x)
        xr = tfm.inverse_transform(y)
        err = np.mean(np.abs(x - xr))
        self.assertLess(err, 1e-2)


class TestWGANGP(unittest.TestCase):
    def test_gp_finite(self):
        critic = Critic(10, [16, 16], use_spectral_norm=False)
        real = torch.randn(32, 10)
        fake = torch.randn(32, 10)
        gp = gradient_penalty(critic, real, fake)
        self.assertTrue(torch.isfinite(gp))
        self.assertGreaterEqual(gp.item(), 0.0)

    def test_critic_minibatch_disc_finite(self):
        critic = Critic(
            10,
            [16, 16],
            use_spectral_norm=False,
            minibatch_discrimination_cfg={"enabled": True, "stat": "mean_abs_diff"},
        )
        x = torch.randn(32, 10)
        y = critic(x)
        self.assertEqual(y.shape, (32,))
        self.assertTrue(torch.isfinite(y).all())


class TestPhysicsExtension(unittest.TestCase):
    def test_species_bounds_hinge(self):
        fake_species = torch.tensor([[0.2, 0.5], [1.2, -0.1]], dtype=torch.float32)
        lo = torch.tensor([0.0, 0.0], dtype=torch.float32)
        hi = torch.tensor([1.0, 1.0], dtype=torch.float32)
        loss, violate = species_bounds_hinge_loss(fake_species, lo, hi, use_hinge=True)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)
        self.assertGreater(violate.item(), 0.0)


class TestGeneratorExtension(unittest.TestCase):
    def test_condition_encoder_shape(self):
        g = Generator(
            latent_dim=8,
            condition_dim=2,
            output_dim=10,
            hidden_dims=[16, 16],
            condition_encoder_cfg={"enabled": True, "hidden_dims": [8], "activation": "gelu"},
        )
        z = torch.randn(4, 8)
        c = torch.randn(4, 2)
        y = g(z, c)
        self.assertEqual(y.shape, (4, 10))
        self.assertTrue(torch.isfinite(y).all())


class TestPairedData(unittest.TestCase):
    def test_paired_bundle_shape(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            x = np.abs(np.random.randn(64, 11).astype(np.float32)) + 1e-4
            y = np.abs(np.random.randn(64, 8).astype(np.float32)) + 1e-4
            x_path = tdp / "x.npy"
            y_path = tdp / "y.npy"
            np.save(x_path, x)
            np.save(y_path, y)
            bundle = create_paired_data_bundle(
                input_npy_path=str(x_path),
                target_npy_path=str(y_path),
                batch_size=16,
                val_ratio=0.2,
                seed=42,
                subset_size=32,
                use_bct=True,
                standardize=True,
            )
            self.assertEqual(bundle.input_dim, 11)
            self.assertEqual(bundle.target_dim, 8)
            bx, by = next(iter(bundle.train_loader))
            self.assertEqual(bx.shape[1], 11)
            self.assertEqual(by.shape[1], 8)


class TestOracle(unittest.TestCase):
    def test_oracle_stub_output_dim(self):
        x = torch.randn(10, 11)
        y, source = get_true_prediction(x, target_dim=8)
        self.assertEqual(y.shape, (10, 8))
        self.assertTrue(isinstance(source, str))

    def test_oracle_cantera_from_10d_input(self):
        x = torch.abs(torch.randn(4, 10))
        x[:, 0] = x[:, 0] * 1000 + 800
        x[:, 1:] = x[:, 1:] / x[:, 1:].sum(dim=1, keepdim=True)
        y, _ = get_true_prediction(x, target_dim=8, mechanism_path="mechanism/Burke2012_s9r23.yaml", time_step=1e-7)
        self.assertEqual(y.shape, (4, 8))
        self.assertTrue(torch.isfinite(y).all())


class TestPlot(unittest.TestCase):
    def test_plot_distribution_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            real = np.abs(np.random.randn(256, 10).astype(np.float32))
            gen = np.abs(np.random.randn(256, 10).astype(np.float32))
            res = plot_distribution_comparison(real, gen, out)
            self.assertTrue(Path(res["stats_file"]).exists())
            self.assertTrue(Path(res["hist_plot"]).exists())
            self.assertTrue(Path(res["pca_plot"]).exists())


if __name__ == "__main__":
    unittest.main()
