"""
Файл предварительной загрузки моделей из huggingface
Без него каждый запрос transcribe будет подгружать все файлы заново
Запустить ровно один раз для
"""

import os
import sys
import whisperx

sys.path.insert(0, os.path.dirname(__file__))

os.environ['HF_HOME'] = os.path.expanduser('~/.cache/huggingface')
os.environ['TORCH_HOME'] = os.path.expanduser('~/.cache/torch')
os.environ['PYANNOTE_CACHE'] = os.path.expanduser('~/.cache/torch/pyannote')

os.makedirs(os.environ['HF_HOME'], exist_ok=True)
os.makedirs(os.environ['TORCH_HOME'], exist_ok=True)
os.makedirs(os.environ['PYANNOTE_CACHE'], exist_ok=True)




try:
    model = whisperx.load_model(
        "base",
        device="cpu",
        compute_type="int8",
        download_root=os.environ['TORCH_HOME']
    )
    model_en, metadata = whisperx.load_align_model(
        language_code="en",
        device="cpu",
        model_dir=os.environ['TORCH_HOME']
    )
    model_ru, metadata = whisperx.load_align_model(
        language_code="ru",
        device="cpu",
        model_dir=os.environ['TORCH_HOME']
    )
    
except Exception as e:
    print(f"\n  ERROR: {e}")