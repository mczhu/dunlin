"""Check explicit and automatic device choices without accelerator hardware."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from dunlin.device import select_device, synchronize


class DeviceTests(unittest.TestCase):
    def test_auto_preserves_mps_cuda_cpu_priority_and_dtype(self) -> None:
        with (
            patch("dunlin.device.torch.backends.mps.is_available", return_value=True),
            patch("dunlin.device.torch.cuda.is_available", return_value=True),
        ):
            self.assertEqual(select_device(), ("mps", torch.float16))
        with (
            patch("dunlin.device.torch.backends.mps.is_available", return_value=False),
            patch("dunlin.device.torch.cuda.is_available", return_value=True),
        ):
            self.assertEqual(select_device(), ("cuda", torch.float16))
        with (
            patch("dunlin.device.torch.backends.mps.is_available", return_value=False),
            patch("dunlin.device.torch.cuda.is_available", return_value=False),
        ):
            self.assertEqual(select_device(), ("cpu", torch.float32))

    def test_explicit_choices_and_invalid_requests(self) -> None:
        self.assertEqual(select_device("cpu", "bfloat16"), ("cpu", torch.bfloat16))
        with patch("dunlin.device.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "CUDA.*unavailable"):
                select_device("cuda:0")
        with patch("dunlin.device.torch.backends.mps.is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "MPS.*unavailable"):
                select_device("mps")
        with (
            patch("dunlin.device.torch.cuda.is_available", return_value=True),
            patch("dunlin.device.torch.cuda.device_count", return_value=2),
        ):
            self.assertEqual(select_device("cuda:1", "float16"), ("cuda:1", torch.float16))
            with self.assertRaisesRegex(ValueError, "CUDA device 2 is unavailable"):
                select_device("cuda:2")
            with self.assertRaisesRegex(ValueError, "Invalid CUDA device index"):
                select_device("cuda:-1")
        with self.assertRaisesRegex(ValueError, "Unknown device"):
            select_device("gpu")
        with self.assertRaisesRegex(ValueError, "Unknown dtype"):
            select_device("cpu", "float8")

    def test_synchronize_targets_selected_cuda_index(self) -> None:
        with (
            patch("dunlin.device.torch.cuda.synchronize") as cuda_sync,
            patch("dunlin.device.torch.mps.synchronize") as mps_sync,
        ):
            synchronize("cuda:1")
            cuda_sync.assert_called_once_with("cuda:1")
            synchronize("mps")
            mps_sync.assert_called_once_with()
            synchronize("cpu")
            self.assertEqual(cuda_sync.call_count, 1)


if __name__ == "__main__":
    unittest.main()
