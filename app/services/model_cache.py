import torch
import typing
import omegaconf

torch.serialization.add_safe_globals([
    omegaconf.listconfig.ListConfig,
    omegaconf.base.ContainerMetadata,
    omegaconf.dictconfig.DictConfig,
    typing.Any,
    dict,
    list,
    bool,
    int,
    float,
    str,
    type(None),
])
import whisperx
import logging
import os
from app.core.config import settings

logger = logging.getLogger(__name__)

os.environ.setdefault('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
os.environ.setdefault('TORCH_HOME', os.path.expanduser('~/.cache/torch'))

_whisper_model = None
_align_models = {}


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        logger.info("Loading Whisper model from disk...")
        _whisper_model = whisperx.load_model(
            settings.WHISPER_MODEL, device="cpu", compute_type=settings.COMPUTE_TYPE,
            download_root=os.environ['TORCH_HOME']
        )
        logger.info("Whisper model ready!")
    return _whisper_model


def get_align_model(language_code):
    global _align_models
    if language_code not in _align_models:
        _align_models[language_code] = whisperx.load_align_model(
            language_code=language_code, device="cpu",
            model_dir=os.environ['TORCH_HOME']
        )
    return _align_models[language_code]