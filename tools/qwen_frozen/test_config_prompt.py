"""Training/checkpoint config dumps must retain the exact native prompt."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mmcv import Config


class PromptRoundtrip(unittest.TestCase):
    def test_quoted_multiline_prompt_roundtrip(self):
        text = "The ego vehicle's lane.\n\nPreserve \\ and \"quoted\" text."
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'source.py'
            source.write_text('prompt = '+repr(text)+'\nmodel = dict(prompt=prompt)\n')
            cfg = Config.fromfile(str(source))
            path = str(Path(folder)/'config.py'); cfg.dump(path)
            loaded = Config.fromfile(path)
            self.assertEqual(loaded.prompt, text)
            self.assertEqual(loaded.model.prompt, text)

    def test_real_training_config_roundtrip(self):
        cfg = Config.fromfile(str(ROOT/'adzoo/drivetransformer/configs/drivetransformer/drivetransformer_qwen08_frozensem_10ep.py'))
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder)/'config.py'); cfg.dump(path)
            loaded = Config.fromfile(path)
            self.assertEqual(loaded.model.img_backbone.prompt, cfg.model.img_backbone.prompt)
            self.assertEqual(loaded.model, cfg.model)
            self.assertEqual(loaded.data, cfg.data)


if __name__ == '__main__': unittest.main()
