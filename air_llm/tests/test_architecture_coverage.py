"""Architecture coverage: models whose decoder is not under ``model.*``.

AirLLM's default streaming layout assumes ``model.embed_tokens`` / ``model.layers`` /
``model.norm``. GPT-2, GPT-NeoX, OPT, Falcon and BLOOM keep the same decoder behind different
attribute names, and two of them have no trunk-level final norm at all. These tests build tiny
checkpoints for each family, stream them through ``AutoModel``, and require the output to match
plain ``transformers`` token for token.

They are the regression net for the automatic layout discovery: a change that breaks the
``transformer.h`` or ``model.decoder.layers`` convention shows up here, not in a user report.
"""

import json
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

_AIRLLM_DIR = Path(__file__).resolve().parents[1] / 'airllm'
if 'airllm' not in sys.modules:
    _pkg = types.ModuleType('airllm')
    _pkg.__path__ = [str(_AIRLLM_DIR)]
    sys.modules['airllm'] = _pkg

from airllm.utils import find_decoder_layout, layout_from_config  # noqa: E402
from airllm.airllm_base import AirLLMBaseModel  # noqa: E402,F401


VOCAB = 512


def write_tokenizer(root):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(["The capital of France is Paris. " * 40], trainers.BpeTrainer(
        vocab_size=VOCAB, special_tokens=["<unk>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok.save(str(root / 'tokenizer.json'))
    (root / 'tokenizer_config.json').write_text(json.dumps({
        "tokenizer_class": "TokenizersBackend", "bos_token": "<unk>",
        "eos_token": "<unk>", "unk_token": "<unk>"}))


def write_checkpoint(root, config, model):
    save_file({k: v.contiguous().clone() for k, v in model.state_dict().items()},
              str(root / 'model.safetensors'))
    data = json.loads(config.to_json_string())
    data['model_type'] = config.model_type
    (root / 'config.json').write_text(json.dumps(data, indent=2))
    write_tokenizer(root)


def _configs():
    """One tiny config per verified family.

    The point of this matrix is breadth: each entry is a decoder whose container names differ from
    Llama's, so discovery either resolves them or the model is unreachable. Adding a family here is
    the cheapest way to guarantee the generic engine keeps working for it.
    """
    from transformers import (BloomConfig, CohereConfig, FalconConfig, GPT2Config,
                              GPTBigCodeConfig, GPTJConfig, GPTNeoXConfig, GemmaConfig,
                              MistralConfig, OPTConfig, OlmoConfig, PhiConfig, Qwen2Config,
                              StableLmConfig)
    common = dict(num_hidden_layers=2, num_attention_heads=4)
    hidden = dict(hidden_size=64, intermediate_size=128, num_key_value_heads=4,
                  max_position_embeddings=128)
    return {
        'gpt2': GPT2Config(vocab_size=VOCAB, n_positions=128, n_embd=64, n_layer=2, n_head=4,
                           bos_token_id=0, eos_token_id=1),
        'gpt_neox': GPTNeoXConfig(vocab_size=VOCAB, intermediate_size=128,
                                  max_position_embeddings=128, **common, hidden_size=64),
        'opt': OPTConfig(vocab_size=VOCAB, hidden_size=64, ffn_dim=128,
                         max_position_embeddings=128, word_embed_proj_dim=64, **common),
        'bloom': BloomConfig(vocab_size=VOCAB, hidden_size=64, n_layer=2, n_head=4),
        'falcon': FalconConfig(vocab_size=VOCAB, hidden_size=64, num_kv_heads=4,
                               new_decoder_architecture=False, **common),
        # Same ``model.*`` trunk as Llama but different block internals, norms and MLP names.
        'mistral': MistralConfig(vocab_size=VOCAB, **common, **hidden),
        'qwen2': Qwen2Config(vocab_size=VOCAB, **common, **hidden),
        'gemma': GemmaConfig(vocab_size=VOCAB, head_dim=16, **common, **hidden),
        'phi': PhiConfig(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, **common),
        'stablelm': StableLmConfig(vocab_size=VOCAB, **common, **hidden),
        'cohere': CohereConfig(vocab_size=VOCAB, **common, **hidden),
        'olmo': OlmoConfig(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, **common),
        # The ``transformer.h`` convention shared by GPT-J and StarCoder.
        'gptj': GPTJConfig(vocab_size=VOCAB, n_embd=64, n_layer=2, n_head=4, rotary_dim=8),
        'gptbigcode': GPTBigCodeConfig(vocab_size=VOCAB, n_embd=64, n_layer=2, n_head=4),
    }


class TestLayoutDiscovery(unittest.TestCase):
    """Discovery is pure (config-only), so it can be asserted without any weights."""

    EXPECTED = {
        'gpt2': {'embed': 'transformer.wte', 'layers': 'transformer.h', 'norm': 'transformer.ln_f'},
        'gpt_neox': {'embed': 'gpt_neox.embed_in', 'layers': 'gpt_neox.layers',
                     'norm': 'gpt_neox.final_layer_norm'},
        'opt': {'embed': 'model.decoder.embed_tokens', 'layers': 'model.decoder.layers',
                'norm': 'model.decoder.final_layer_norm'},
        'bloom': {'embed': 'transformer.word_embeddings', 'layers': 'transformer.h',
                  'norm': 'transformer.ln_f'},
        'falcon': {'embed': 'transformer.word_embeddings', 'layers': 'transformer.h',
                   'norm': 'transformer.ln_f'},
        'mistral': {'embed': 'model.embed_tokens', 'layers': 'model.layers', 'norm': 'model.norm'},
        'qwen2': {'embed': 'model.embed_tokens', 'layers': 'model.layers', 'norm': 'model.norm'},
        'gemma': {'embed': 'model.embed_tokens', 'layers': 'model.layers', 'norm': 'model.norm'},
        'phi': {'embed': 'model.embed_tokens', 'layers': 'model.layers',
                'norm': 'model.final_layernorm'},
        'stablelm': {'embed': 'model.embed_tokens', 'layers': 'model.layers', 'norm': 'model.norm'},
        'cohere': {'embed': 'model.embed_tokens', 'layers': 'model.layers', 'norm': 'model.norm'},
        'olmo': {'embed': 'model.embed_tokens', 'layers': 'model.layers', 'norm': 'model.norm'},
        'gptj': {'embed': 'transformer.wte', 'layers': 'transformer.h', 'norm': 'transformer.ln_f'},
        'gptbigcode': {'embed': 'transformer.wte', 'layers': 'transformer.h',
                       'norm': 'transformer.ln_f'},
    }

    def test_each_family_resolves_its_own_container_names(self):
        for model_type, config in _configs().items():
            layout = layout_from_config(config)
            self.assertIsNotNone(layout, f"no layout discovered for {model_type}")
            expected = self.EXPECTED[model_type]
            self.assertEqual(layout['embed'], expected['embed'], model_type)
            self.assertEqual(layout['layer_prefix'], expected['layers'], model_type)
            self.assertEqual(layout['norm'], expected['norm'], model_type)

    def test_bloom_post_embedding_norm_becomes_its_own_stage(self):
        """BLOOM normalizes before the first block; that module must not be left on meta."""
        layout = layout_from_config(_configs()['bloom'])
        self.assertIn('transformer.word_embeddings_layernorm',
                      layout.get('prefix_modules', []))

    def test_gpt2_position_table_is_resident(self):
        layout = layout_from_config(_configs()['gpt2'])
        self.assertIn('transformer.wpe', layout.get('resident', []))

    def test_llama_style_models_need_no_discovery(self):
        from transformers import LlamaConfig
        found = find_decoder_layout.__doc__  # keep import linters honest
        self.assertIsNotNone(found)
        layout = layout_from_config(LlamaConfig(vocab_size=VOCAB, hidden_size=64,
                                                num_hidden_layers=2, num_attention_heads=4,
                                                num_key_value_heads=4, intermediate_size=128))
        self.assertEqual(layout['embed'], 'model.embed_tokens')
        self.assertEqual(layout['norm'], 'model.norm')
        # The default layout already points here, so nothing extra should be invented.
        self.assertFalse(layout.get('prefix_modules'))


class TestStreamingMatchesTransformers(unittest.TestCase):
    """End-to-end: stream a tiny checkpoint of each family and compare generated tokens."""

    @classmethod
    def setUpClass(cls):
        from transformers import AutoModelForCausalLM
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.paths = {}
        for model_type, config in _configs().items():
            path = cls.root / model_type
            path.mkdir()
            write_checkpoint(path, config, AutoModelForCausalLM.from_config(config))
            cls.paths[model_type] = path
        cls.ids = torch.tensor([[5, 9, 12, 44, 7, 8]])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _compare(self, model_type, codec=None):
        from transformers import AutoModelForCausalLM

        path = self.paths[model_type]
        shutil.rmtree(path / 'splitted_model', ignore_errors=True)
        kwargs = {'codec': codec} if codec else {}
        ref = AutoModelForCausalLM.from_pretrained(str(path))
        ref.eval()
        with torch.no_grad():
            expected = ref.generate(self.ids, max_new_tokens=4, use_cache=True,
                                    return_dict_in_generate=True, do_sample=False)
        model = AirLLMBaseModel(str(path), full_load=False, **kwargs)
        got = model.generate(self.ids, max_new_tokens=4, use_cache=True,
                             return_dict_in_generate=True, do_sample=False)
        self.assertEqual(expected.sequences[0].tolist(), got.sequences[0].tolist(), model_type)

    def test_every_family_streams_identically(self):
        """The whole matrix, so a new family cannot be added untested."""
        for model_type in _configs():
            with self.subTest(model_type=model_type):
                self._compare(model_type)

    def test_every_family_streams_identically_with_a_lossless_codec(self):
        """A lossless shard must not change a single token, on any architecture.

        This is the contract that lets a user shrink a checkpoint on disk without reasoning about
        accuracy at all -- and it has to hold for the generic path, not just Llama.
        """
        for model_type in _configs():
            with self.subTest(model_type=model_type):
                self._compare(model_type, codec='lossless')


if __name__ == '__main__':
    unittest.main()
