from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.models.speakers import Speaker
from app.api.v1.dependencies import get_current_active_user, get_session
from app.services.speakers import get_user_speakers, rename_speaker

router = APIRouter(prefix="/speakers", tags=["speakers"])


@router.get("/")
async def list_speakers(
    current_user: User = Depends(get_current_active_user)
):
    return await get_user_speakers(current_user.id)


@router.put("/{speaker_id}")
async def update_speaker(
    speaker_id: int,
    name: str,
    current_user: User = Depends(get_current_active_user)
):
    success = await rename_speaker(speaker_id, name, current_user.id)
    if not success:
        raise HTTPException(status_code=404, detail="Speaker not found")
    return {"message": "Speaker renamed"}


@router.delete("/{speaker_id}")
async def delete_speaker(
    speaker_id: int,
    current_user: User = Depends(get_current_active_user),
    session: AsyncSession = Depends(get_session)
):
    from sqlalchemy import select, delete
    from app.models.speakers import SegmentSpeaker
    
    speaker = await session.get(Speaker, speaker_id)
    if not speaker or speaker.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Speaker not found")
    
    await session.execute(
        delete(SegmentSpeaker).where(SegmentSpeaker.speaker_id == speaker_id)
    )
    await session.delete(speaker)
    await session.commit()
    
    return {"message": "Speaker deleted"}