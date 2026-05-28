import os
import uuid
import aiofiles
from pathlib import Path
from fastapi import UploadFile, HTTPException, status
from app.core.config import settings


async def save_upload_file(upload_file: UploadFile, user_id: int) -> str:
    user_dir = Path(settings.UPLOAD_DIR).resolve() / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    
    file_extension = Path(upload_file.filename).suffix if upload_file.filename else ".wav"
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = user_dir / unique_filename
    
    async with aiofiles.open(file_path, 'wb') as out_file:
        while content := await upload_file.read(1024 * 1024):
            await out_file.write(content)
    
    return str(file_path.resolve())


def validate_audio_file(upload_file: UploadFile) -> None:
    if not upload_file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided"
        )
    
    allowed_extensions = {'.wav', '.mp3', '.m4a', '.ogg', '.flac', '.aac', '.mp4'}
    file_ext = Path(upload_file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format. Allowed: {', '.join(allowed_extensions)}"
        )