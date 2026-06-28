import asyncio
import pickle
import numpy as np
import librosa
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy import select
from app.core.database import async_session_maker
from app.models.speakers import Speaker
from sklearn.metrics.pairwise import cosine_similarity

_executor = ThreadPoolExecutor(max_workers=2)


def serialize_voiceprint(mfcc_vector):
    return pickle.dumps(mfcc_vector.astype(np.float32))


def deserialize_voiceprint(data):
    return pickle.loads(data)


def extract_mfcc_from_audio(audio_array, sample_rate):
    rms = np.sqrt(np.mean(audio_array ** 2))
    if rms > 0:
        audio_array = audio_array / rms
    
    mfcc = librosa.feature.mfcc(y=audio_array, sr=sample_rate, n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    combined = np.vstack([mfcc, delta, delta2])
    return np.mean(combined, axis=1).astype(np.float32)


def compare_voiceprints(vp1, vp2):
    return float(cosine_similarity(vp1.reshape(1, -1), vp2.reshape(1, -1))[0][0])


def update_voiceprint(old_vp, new_vp, sample_count):
    alpha = 1.0 / (sample_count + 1)
    return (1 - alpha) * old_vp + alpha * new_vp


def process_all_segments_sync(full_audio, sample_rate, segments, existing_speakers, similarity_threshold):
    new_speakers = []
    segment_results = []
    updated_speakers = []

    known = []
    for spk in existing_speakers:
        known.append({
            'db_id': spk.id,
            'name': spk.name,
            'vp': deserialize_voiceprint(spk.voiceprint),
            'cnt': 1
        })

    

    for seg in segments:
        a = int(seg['start'] * sample_rate)
        b = int(seg['end'] * sample_rate)
        chunk = full_audio[a:b]

        new_vp = extract_mfcc_from_audio(chunk, sample_rate)
        for k in known:
            s = compare_voiceprints(new_vp, k['vp'])
            print(f"  vs {k['name']}: {s:.3f} {'✓' if s >= similarity_threshold else ''}")

        best = None
        best_score = 0.0

        for k in known:
            s = compare_voiceprints(new_vp, k['vp'])
            if s > best_score and s >= similarity_threshold:
                best_score = s
                best = k

        if best is not None:
            best['vp'] = update_voiceprint(best['vp'], new_vp, best['cnt'])
            best['cnt'] += 1
            updated_speakers.append(best)
            segment_results.append({
                'start': seg['start'],
                'end': seg['end'],
                'speaker': best['name'],
                'db_id': best['db_id']
            })
        else:
            ns = {
                'db_id': None,
                'name': f"Speaker {len(known) + 1}",
                'vp': new_vp,
                'cnt': 1,
                'vp_bytes': serialize_voiceprint(new_vp)
            }
            new_speakers.append(ns)
            known.append(ns)
            segment_results.append({
                'start': seg['start'],
                'end': seg['end'],
                'speaker': ns['name'],
                'db_id': None,
                'new': ns
            })

    return segment_results, new_speakers, updated_speakers


async def process_diarization(user_id, wav_path, segments, similarity_threshold=0.9):
    loop = asyncio.get_event_loop()
    import soundfile as sf

    full_audio, sr = await loop.run_in_executor(_executor, lambda: sf.read(wav_path, dtype='float32'))

    async with async_session_maker() as session:
        result = await session.execute(select(Speaker).where(Speaker.user_id == user_id))
        existing = result.scalars().all()

    results, new_speakers, updated = await loop.run_in_executor(
        _executor, process_all_segments_sync,
        full_audio, sr, segments, existing, similarity_threshold
    )

    async with async_session_maker() as session:
        for ns in new_speakers:
            spk = Speaker(user_id=user_id, name=ns['name'], voiceprint=ns['vp_bytes'])
            session.add(spk)
            await session.flush()
            for r in results:
                if r.get('new') is ns:
                    r['db_id'] = spk.id

        for u in updated:
            spk = await session.get(Speaker, u['db_id'])
            if spk:
                spk.voiceprint = serialize_voiceprint(u['vp'])
                session.add(spk)

        await session.commit()

    return results


async def rename_speaker(speaker_id, new_name, user_id):
    async with async_session_maker() as session:
        result = await session.execute(
            select(Speaker).where(Speaker.id == speaker_id, Speaker.user_id == user_id)
        )
        spk = result.scalar_one_or_none()
        if spk:
            spk.name = new_name
            session.add(spk)
            await session.commit()
            return True
    return False


async def get_user_speakers(user_id):
    async with async_session_maker() as session:
        print(user_id)
        result = await session.execute(select(Speaker).where(Speaker.user_id == user_id))
        speakers = result.scalars().all()
        print(speakers)
        return [
            {"id": s.id, "name": s.name, "created_at": s.created_at.isoformat()}
            for s in speakers
        ]