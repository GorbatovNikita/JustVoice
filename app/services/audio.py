from sqlalchemy import select
from sqlalchemy.orm import selectinload
from app.core.database import async_session_maker
from app.models.transcription import TranscriptionTask, TranscriptionSegment
from app.services.transcription_queue import add_to_queue


async def process_audio_task(task_id: int):
    await add_to_queue(task_id)


async def get_transcription_task(task_id: int, user_id: int):
    async with async_session_maker() as session:
        query = select(TranscriptionTask).where(
            TranscriptionTask.id == task_id,
            TranscriptionTask.user_id == user_id
        ).options(selectinload(TranscriptionTask.segments))
        result = await session.execute(query)
        return result.scalar_one_or_none()


async def get_user_transcriptions(user_id: int):
    async with async_session_maker() as session:
        query = select(TranscriptionTask).where(
            TranscriptionTask.user_id == user_id
        ).order_by(TranscriptionTask.created_at.desc())
        result = await session.execute(query)
        tasks = result.scalars().all()
        
        result_list = []
        for task in tasks:
            count_query = select(TranscriptionSegment).where(
                TranscriptionSegment.task_id == task.id
            )
            count_result = await session.execute(count_query)
            segments_count = len(count_result.scalars().all())
            
            result_list.append({
                "id": task.id,
                "original_filename": task.original_filename,
                "display_name": task.display_name,
                "status": task.status,
                "language": task.language,
                "duration": task.duration,
                "topic": task.topic,
                "segments_count": segments_count,
                "created_at": task.created_at.isoformat() if task.created_at else None
            })
        
        return result_list
    