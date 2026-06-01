"""Regression tests for MiMoV2 NextN hybrid SWA layer-id selection."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.configs.model_config import (
    get_first_swa_layer_id_from_hybrid_pattern,
    get_hybrid_layer_ids,
)
from sglang.srt.models.mimo_v2_nextn import MiMoV2ModelNextN


class TestMiMoV2NextN(CustomTestCase):
    def test_get_first_swa_layer_id_from_hybrid_pattern(self):
        cfg = SimpleNamespace(hybrid_layer_pattern=[0, 0, 1, 1])
        self.assertEqual(get_first_swa_layer_id_from_hybrid_pattern(cfg), 2)
        with self.assertRaisesRegex(ValueError, "hybrid_layer_pattern"):
            get_first_swa_layer_id_from_hybrid_pattern(SimpleNamespace())
        with self.assertRaisesRegex(ValueError, "at least one SWA layer"):
            get_first_swa_layer_id_from_hybrid_pattern(
                SimpleNamespace(hybrid_layer_pattern=[0, 0])
            )

    def test_mimo_v2_mtp_hybrid_ids_use_first_swa_layer(self):
        cfg = SimpleNamespace(
            num_hidden_layers=4,
            hybrid_layer_pattern=[0, 1, 1, 0],
        )
        self.assertEqual(get_hybrid_layer_ids(["MiMoV2MTP"], cfg), ([1], []))

    @patch("sglang.srt.models.mimo_v2_nextn.MiMoV2MTPLayer")
    @patch("sglang.srt.models.mimo_v2_nextn.RMSNorm")
    @patch("sglang.srt.models.mimo_v2_nextn.VocabParallelEmbedding")
    def test_model_nextn_uses_first_swa_layer_for_mtp_attention(
        self,
        _mock_embed_tokens,
        _mock_rmsnorm,
        mock_mtp_layer,
    ):
        cfg = SimpleNamespace(
            vocab_size=32,
            hidden_size=16,
            layernorm_epsilon=1e-5,
            hybrid_layer_pattern=[0, 1, 1, 1],
        )

        MiMoV2ModelNextN(cfg)

        self.assertEqual(mock_mtp_layer.call_args.args[1], 1)


if __name__ == "__main__":
    unittest.main()
