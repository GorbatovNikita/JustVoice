import torch
import torch.serialization

try:
    import omegaconf.listconfig
    torch.serialization.add_safe_globals([omegaconf.listconfig.ListConfig])
except:
    pass


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.endpoints import auth, audio, users, speakers


app = FastAPI(
    title="FastAPI JWT Auth Demo",
    description="Пример реализации JWT авторизации на FastAPI",
    version="1.0.0"
)

# CORS для разработки
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Регистрируем роутеры
app.include_router(auth.router, prefix="/api/v1")
app.include_router(audio.router, prefix="/api/v1")
app.include_router(users.router, prefix="/api/v1")
app.include_router(speakers.router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "FastAPI JWT Auth Demo. Go to /docs for API documentation"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)