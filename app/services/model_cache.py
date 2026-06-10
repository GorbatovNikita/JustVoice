import torch
import warnings
import os
import whisperx
import logging

logger = logging.getLogger(__name__)

os.environ.setdefault('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
os.environ.setdefault('TORCH_HOME', os.path.expanduser('~/.cache/torch'))

_whisper_model = None
_align_models = {}

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        logger.info("Loading Whisper model...")
        from app.core.config import settings
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            old_load = torch.load
            def patched_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return old_load(*args, **kwargs)
            torch.load = patched_load
            
            _whisper_model = whisperx.load_model(
                settings.WHISPER_MODEL, device="cpu", compute_type=settings.COMPUTE_TYPE,
                download_root=os.environ['TORCH_HOME']
            )
            
            torch.load = old_load
        
        logger.info("Whisper model ready!")
    return _whisper_model


def get_align_model(language_code):
    global _align_models
    if language_code not in _align_models:
        logger.info(f"Loading alignment model for {language_code}...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            old_load = torch.load
            def patched_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return old_load(*args, **kwargs)
            torch.load = patched_load
            
            _align_models[language_code] = whisperx.load_align_model(
                language_code=language_code, device="cpu",
                model_dir=os.environ['TORCH_HOME']
            )
            
            torch.load = old_load
        
        logger.info(f"Alignment model for {language_code} ready!")
    return _align_models[language_code]