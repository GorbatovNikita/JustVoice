import asyncio
import logging
from collections import deque
import typing
if not hasattr(typing, '_UnionGenericAlias'):
    typing._UnionGenericAlias = getattr(typing, 'Union')

logger = logging.getLogger(__name__)

_transcription_queue = deque()
_is_processing = False
_queue_lock = asyncio.Lock()


async def add_to_queue(task_id: int):
    async with _queue_lock:
        _transcription_queue.append(task_id)
        logger.info(f"Task {task_id} added to queue. Queue size: {len(_transcription_queue)}")
    
    asyncio.create_task(_process_queue())


async def _process_queue():
    global _is_processing
    
    async with _queue_lock:
        if _is_processing:
            return
        if not _transcription_queue:
            return
        _is_processing = True
        task_id = _transcription_queue.popleft()
    
    try:
        await process_single_task(task_id)
    finally:
        async with _queue_lock:
            _is_processing = False
            if _transcription_queue:
                asyncio.create_task(_process_queue())


async def process_single_task(task_id: int):
    import whisperx
    import os
    import tempfile
    import subprocess
    import traceback
    import soundfile as sf
    import numpy as np
    import torch
    
    from app.core.database import async_session_maker
    from app.models.transcription import TranscriptionTask, TranscriptionSegment
    from app.services.model_cache import get_whisper_model, get_align_model
    from app.services.topic_classifier import classify_topic
    from app.services.speakers import process_diarization
    
    logger.info(f"Processing task {task_id}")
    
    async with async_session_maker() as session:
        try:
            task = await session.get(TranscriptionTask, task_id)
            if not task:
                return
            
            file_path = task.file_path
            
            if not os.path.exists(file_path):
                task.status = "failed"
                task.error_message = "File not found"
                await session.commit()
                return
            
            task.status = "processing"
            await session.commit()
            
            try:
                wav_path = tempfile.mktemp(suffix='.wav')
                subprocess.run([
                    'ffmpeg', '-y', '-i', file_path,
                    '-acodec', 'pcm_s16le',
                    '-ar', '16000',
                    '-ac', '1',
                    '-af', 'highpass=f=200,lowpass=f=3000,volume=2.0',
                    wav_path
                ], capture_output=True, text=True)
                
                audio_data, sr = sf.read(wav_path, dtype='float32')
                total_duration = len(audio_data) / sr
                
                loop = asyncio.get_event_loop()
                
                model = await loop.run_in_executor(None, get_whisper_model)
                result = await loop.run_in_executor(None, model.transcribe, wav_path, 16)
                
                language = result["language"]
                
                model_a, metadata = await loop.run_in_executor(None, get_align_model, language)
                aligned = await loop.run_in_executor(
                    None, whisperx.align, result["segments"], model_a, metadata, wav_path, "cpu"
                )
                
                segments_list = aligned.get("segments", [])
                
                if len(segments_list) > 1:
                    try:
                        results = await process_diarization(
                            user_id=task.user_id,
                            wav_path=wav_path,
                            segments=segments_list
                        )
                        
                        for i, seg in enumerate(segments_list):
                            if i < len(results):
                                seg['speaker'] = results[i]['speaker']
                                seg['speaker_id'] = results[i].get('db_id')
                            else:
                                seg['speaker'] = "SPEAKER_00"
                        
                        speaker_count = len(set(
                            r['speaker'] for r in results if r.get('speaker') != 'Unknown'
                        ))
                        logger.info(f"Identified {speaker_count} speakers")
                        
                    except Exception as e:
                        logger.warning(f"Diarization failed: {e}")
                        for seg in segments_list:
                            seg['speaker'] = "SPEAKER_00"
                else:
                    for seg in segments_list:
                        seg['speaker'] = "SPEAKER_00"
                
                os.remove(wav_path)
                
                task.language = language
                task.duration = total_duration
                
                for seg in segments_list:
                    segment = TranscriptionSegment(
                        task_id=task.id,
                        speaker=seg.get("speaker", "SPEAKER_00"),
                        start_time=seg.get("start", 0),
                        end_time=seg.get("end", 0),
                        text=seg.get("text", "").strip(),
                        confidence=seg.get("confidence", None)
                    )
                    session.add(segment)
                    await session.flush()
                    
                    speaker_id = seg.get('speaker_id')
                    if speaker_id is not None:
                        from app.models.speakers import SegmentSpeaker
                        from sqlalchemy import delete
                        await session.execute(
                            delete(SegmentSpeaker).where(SegmentSpeaker.segment_id == segment.id)
                        )
                        link = SegmentSpeaker(segment_id=segment.id, speaker_id=speaker_id)
                        session.add(link)
                
                full_text = " ".join([seg.get("text", "") for seg in segments_list])
                topic = await loop.run_in_executor(None, classify_topic, full_text)
                task.topic = topic
                logger.info(f"Detected topic: {topic}")
                
                task.status = "completed"
                await session.commit()
                logger.info(f"Task {task_id} completed: {total_duration:.1f}s, topic: {topic}")
                
            except Exception as e:
                logger.error(f"Error: {traceback.format_exc()}")
                task.status = "failed"
                task.error_message = str(e)[:500]
                await session.commit()
                
        except Exception as e:
            logger.error(f"DB error: {traceback.format_exc()}")


def get_queue_size():
    return len(_transcription_queue)