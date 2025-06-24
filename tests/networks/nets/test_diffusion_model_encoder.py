import unittest
import torch
from monai.networks import eval_mode
from monai.networks.nets import DiffusionModelEncoder


class TestDiffusionModelEncoder(unittest.TestCase):
    def test_dynamic_linear(self):
        params = {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 4,
            "num_res_blocks": 1,
            "channels": (4, 8),
            "attention_levels": (False, False),
            "norm_num_groups": 4,
        }
        device = "cuda" if torch.cuda.is_available() else "cpu"
        net = DiffusionModelEncoder(**params).to(device)
        x = torch.randn(2, 1, 32, 32, device=device)
        t = torch.tensor([1, 2], device=device)
        with eval_mode(net):
            out = net(x, t)
        self.assertEqual(out.shape, (2, params["out_channels"]))
        self.assertNotEqual(net.out[0].in_features, 4096)


if __name__ == "__main__":
    unittest.main()
