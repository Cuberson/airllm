import unittest

from ..airllm.auto_model import AutoModel


class TestAutoModel(unittest.TestCase):
    """Architecture -> class mapping.

    This used to call ``get_module_class(repo_id)``, which downloads each checkpoint's config
    from the Hub. The mapping itself is pure, so it is tested directly: a unit test should not
    depend on the network (it failed for exactly that reason on an offline runner).
    """

    MAPPING = {
        'LlamaForCausalLM': 'AirLLMBaseModel',
        'Qwen2ForCausalLM': 'AirLLMBaseModel',
        'MistralForCausalLM': 'AirLLMBaseModel',
        'MixtralForCausalLM': 'AirLLMBaseModel',
        'DeepseekV3ForCausalLM': 'AirLLMBaseModel',
        'QWenLMHeadModel': 'AirLLMQWen',
        'InternLMForCausalLM': 'AirLLMInternLM',
        'ChatGLMModel': 'AirLLMChatGLM',
        'ChatGLMForConditionalGeneration': 'AirLLMChatGLM',
        'BaichuanForCausalLM': 'AirLLMBaichuan',
        'BaiChuanForCausalLM': 'AirLLMBaichuan',
        'KimiK3ForConditionalGeneration': 'AirLLMKimiK3',
        'Qwen3_5ForConditionalGeneration': 'AirLLMQwen3_5',
        'Qwen4ExpForConditionalGeneration': 'AirLLMQwen4Exp',
    }

    def test_architecture_maps_to_expected_class(self):
        for arch, expected in self.MAPPING.items():
            got = AutoModel.class_name_for_architecture(arch)
            self.assertEqual(got, expected, f"expecting {expected} for {arch}")

    def test_unknown_architecture_uses_the_generic_streaming_path(self):
        self.assertEqual(AutoModel.class_name_for_architecture('SomeBrandNewForCausalLM'),
                         'AirLLMBaseModel')
        self.assertEqual(AutoModel.class_name_for_architecture(''), 'AirLLMBaseModel')

    def test_cannot_be_instantiated_directly(self):
        with self.assertRaises(EnvironmentError):
            AutoModel()


if __name__ == '__main__':
    unittest.main()
