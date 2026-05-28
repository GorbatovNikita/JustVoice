from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio
import logging
import os

from app.core.storage import save_upload_file, validate_audio_file
from app.models.transcription import TranscriptionTask
from app.models.user import User
from app.schemas.audio import TranscriptionResponse, TranscriptionStatusResponse, TranscriptionUpdateName
from app.services.audio import process_audio_task, get_transcription_task, get_user_transcriptions
from app.api.v1.dependencies import get_current_active_user, get_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/audio", tags=["audio processing"])

@router.put("/transcription/{task_id}/name")
async def rename_transcription(
    task_id: int,
    name_data: TranscriptionUpdateName,
    current_user: User = Depends(get_current_active_user),
    session: AsyncSession = Depends(get_session)
):
    task = await get_transcription_task(task_id, current_user.id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task.display_name = name_data.display_name
    session.add(task)
    await session.commit()
    
    return {"message": "Name updated", "display_name": task.display_name}


@router.delete("/transcription/{task_id}")
async def delete_transcription(
    task_id: int,
    current_user: User = Depends(get_current_active_user),
    session: AsyncSession = Depends(get_session)
):
    task = await get_transcription_task(task_id, current_user.id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task.file_path and os.path.exists(task.file_path):
        os.remove(task.file_path)
    
    await session.delete(task)
    await session.commit()
    
    return {"message": "Transcription deleted"}

@router.get("/file/{task_id}")
async def download_audio_file(
    task_id: int,
    current_user: User = Depends(get_current_active_user)
):
    task = await get_transcription_task(task_id, current_user.id)
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    file_path = task.file_path
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path, media_type="audio/mpeg", 
                       filename=task.original_filename)


@router.post("/transcribe", response_model=TranscriptionStatusResponse, status_code=status.HTTP_202_ACCEPTED)
async def transcribe_audio(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    session: AsyncSession = Depends(get_session)
):
    validate_audio_file(file)
    
    file_path = await save_upload_file(file, current_user.id)
    
    task = TranscriptionTask(
        user_id=current_user.id,
        original_filename=file.filename,
        file_path=file_path,
        status="pending"
    )

    print(task.topic)

    session.add(task)
    await session.flush()
    await session.refresh(task)
    await session.commit()
    
    task_id = task.id
    logger.info(f"Task {task_id} created, starting background processing")
    asyncio.create_task(process_audio_task(task_id))
    
    return TranscriptionStatusResponse(
        id=task_id,
        original_filename=task.original_filename,
        status=task.status,
        created_at=task.created_at
    )


@router.get("/transcriptions", response_model=list[TranscriptionStatusResponse])
async def list_transcriptions(
    current_user: User = Depends(get_current_active_user)
):
    tasks = await get_user_transcriptions(current_user.id)
    return [
        TranscriptionStatusResponse(
            id=task["id"],
            original_filename=task["original_filename"],
            display_name=task.get("display_name"),
            status=task["status"],
            language=task.get("language"),
            topic=task.get('topic'),
            duration=task.get("duration"),
            segments_count=task.get("segments_count", 0),
            created_at=task["created_at"]
        )
        for task in tasks
    ]

@router.post("/transcription/{task_id}/retry", status_code=status.HTTP_202_ACCEPTED)
async def retry_transcription(
    task_id: int,
    current_user: User = Depends(get_current_active_user),
    session: AsyncSession = Depends(get_session)
):
    task = await get_transcription_task(task_id, current_user.id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task.status = "pending"
    task.error_message = None
    task.topic = None
    session.add(task)
    await session.commit()
    
    await process_audio_task(task.id)
    
    return {"message": "Retry started", "task_id": task.id}

@router.get("/transcription/{task_id}", response_model=TranscriptionResponse)
async def get_transcription(
    task_id: int,
    current_user: User = Depends(get_current_active_user)
):
    task = await get_transcription_task(task_id, current_user.id)
    
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transcription task not found"
        )
    
    return TranscriptionResponse(
        id=task.id,
        original_filename=task.original_filename,
        status=task.status,
        language=task.language,
        duration=task.duration,
        segments=[seg for seg in task.segments] if task.segments else [],
        created_at=task.created_at,
        display_name=task.display_name
    )