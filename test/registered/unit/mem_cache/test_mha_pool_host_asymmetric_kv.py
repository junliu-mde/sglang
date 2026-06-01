import torch

from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.memory_pool_host import MHATokenToKVPoolHost
from sglang.srt.utils import is_cuda, is_hip, is_npu, is_xpu
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=8, stage="stage-b", runner_config="1-gpu-small")


class TestMHAHiCacheAsymmetricKV(CustomTestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for MHA host transfer tests.")
        if is_npu() or is_xpu():
            self.skipTest("MHA host transfer tests only support CUDA/ROCm.")
        if not (is_cuda() or is_hip()):
            self.skipTest("CUDA/ROCm not available.")

    def test_direct_backup_supports_asymmetric_kv_dims(self):
        page_size = 1
        layer_num = 2
        size = 8
        device_pool = MHATokenToKVPool(
            size=size,
            page_size=page_size,
            dtype=torch.bfloat16,
            head_num=2,
            head_dim=192,
            v_head_dim=128,
            layer_num=layer_num,
            device="cuda",
            enable_memory_saver=False,
        )
        host_pool = MHATokenToKVPoolHost(
            device_pool=device_pool,
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=page_size,
            layout="layer_first",
            pin_memory=False,
            device="cpu",
            allocator_type="default",
        )

        self.assertEqual(host_pool.k_buffer.shape[-1], 192)
        self.assertEqual(host_pool.v_buffer.shape[-1], 128)

        for layer_id in range(layer_num):
            k_src = torch.arange(
                device_pool.k_buffer[layer_id].numel(),
                device=device_pool.k_buffer[layer_id].device,
                dtype=torch.bfloat16,
            ).view_as(device_pool.k_buffer[layer_id])
            v_src = torch.arange(
                device_pool.v_buffer[layer_id].numel(),
                device=device_pool.v_buffer[layer_id].device,
                dtype=torch.bfloat16,
            ).view_as(device_pool.v_buffer[layer_id])
            device_pool.k_buffer[layer_id].copy_(k_src + layer_id)
            device_pool.v_buffer[layer_id].copy_(v_src + layer_id)

        device_indices = torch.tensor([1, 3, 5], device="cuda", dtype=torch.int64)
        host_indices = torch.tensor([0, 1, 2], device="cpu", dtype=torch.int64)

        host_pool.backup_from_device_all_layer(
            device_pool,
            host_indices,
            device_indices,
            io_backend="direct",
        )

        for layer_id in range(layer_num):
            expected_k = device_pool.k_buffer[layer_id][device_indices].cpu()
            expected_v = device_pool.v_buffer[layer_id][device_indices].cpu()
            got_k = host_pool.k_buffer[layer_id][host_indices]
            got_v = host_pool.v_buffer[layer_id][host_indices]
            self.assertTrue(torch.equal(got_k, expected_k))
            self.assertTrue(torch.equal(got_v, expected_v))

        k_page, v_page = host_pool.get_data_page(0, flat=False)
        self.assertEqual(k_page.shape, (layer_num, page_size, 2, 192))
        self.assertEqual(v_page.shape, (layer_num, page_size, 2, 128))


if __name__ == "__main__":
    import unittest

    unittest.main()
