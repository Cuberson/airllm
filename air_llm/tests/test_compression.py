import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch

HAVE_BNB = importlib.util.find_spec('bitsandbytes') is not None

# Other tests in this suite install a lightweight ``airllm`` namespace package in sys.modules so
# they can import submodules without the full package (and without its optional dependencies).
# Reuse it when present instead of relying on the installed package layout.
if 'airllm' not in sys.modules:
    _pkg = types.ModuleType('airllm')
    _pkg.__path__ = [str(Path(__file__).resolve().parents[1] / 'airllm')]
    sys.modules['airllm'] = _pkg

from airllm.utils import compress_layer_state_dict, uncompress_layer_state_dict

# bitsandbytes ships CPU kernels for both schemes, so this exercises the real quantize /
# dequantize round trip wherever the tests run. It used to hardcode .cuda(), which made the
# suite unrunnable without an NVIDIA GPU.
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


@unittest.skipUnless(HAVE_BNB, "compression needs bitsandbytes")
class TestCompression(unittest.TestCase):
    def test_should_compress_uncompress(self):
        #torch.manual_seed(0)
        a0 = torch.normal(0, 1, (32, 128), dtype=torch.float16).to(DEVICE)
        a1 = torch.normal(0, 1, (32, 128), dtype=torch.float16).to(DEVICE)

        a_state_dict = {'a0':a0, 'a1':a1}

        loss_fn = torch.nn.MSELoss()

        for iloop in range(10):
            for compression in [None, '4bit', '8bit']:
                b = compress_layer_state_dict(a_state_dict, compression, device=DEVICE)

                if iloop < 2:
                    print(f"for compression {compression}, compressed to: { {k:v.shape for k,v in b.items()} }")

                aa = uncompress_layer_state_dict(b, device=DEVICE)

                for k in aa.keys():

                    if compression is None:
                        self.assertTrue(torch.equal(aa[k], a_state_dict[k]))
                    else:
                        RMSE_loss = torch.sqrt(loss_fn(aa[k].float(), a_state_dict[k].float())).detach().cpu().item()
                        print(f"compression {compression} loss: {RMSE_loss}")
                        self.assertLess(RMSE_loss, 0.1)