import asyncio
import json
import os
from contextlib import suppress
from pathlib import Path
from urllib.parse import urlencode

import websockets
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse

load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
DEEPGRAM_LISTEN_URL = "wss://api.deepgram.com/v1/listen"
PUBLIC_DIR = Path(__file__).parent / "public"
KEEPALIVE_SECONDS = 8
FLUSH_TIMEOUT_SECONDS = 5

app = FastAPI(title="Voice Agent")


@app.get("/")
async def index():
    # Vercel serves public/ from its CDN, so this route is mainly for local runs.
    return FileResponse(PUBLIC_DIR / "index.html")


@app.get("/api/health")
async def health():
    return {"ok": True, "deepgram_key_set": bool(DEEPGRAM_API_KEY)}


def deepgram_url() -> str:
    # The browser sends WebM/Opus from MediaRecorder, so no encoding params are needed.
    params = {
        "model": "nova-3",
        "smart_format": "true",
        "punctuate": "true",
        "interim_results": "true",
        "endpointing": "300",
    }
    return f"{DEEPGRAM_LISTEN_URL}?{urlencode(params)}"


async def forward_audio(browser: WebSocket, dg) -> None:
    """Relay binary audio from the browser to Deepgram until the browser stops."""
    while True:
        message = await browser.receive()
        if message["type"] == "websocket.disconnect":
            break
        if message.get("bytes"):
            await dg.send(message["bytes"])
        elif message.get("text") == "stop":
            break
    # Ask Deepgram to flush any pending results and close the stream.
    with suppress(Exception):
        await dg.send(json.dumps({"type": "CloseStream"}))


async def forward_transcripts(dg, browser: WebSocket) -> None:
    """Relay Deepgram transcript results to the browser."""
    async for raw in dg:
        message = json.loads(raw)
        if message.get("type") != "Results":
            continue
        text = message["channel"]["alternatives"][0].get("transcript", "")
        if not text:
            continue
        await browser.send_json(
            {
                "type": "transcript",
                "text": text,
                "is_final": message.get("is_final", False),
                "speech_final": message.get("speech_final", False),
            }
        )


async def keep_alive(dg) -> None:
    """Keep the Deepgram connection open during pauses in speech."""
    while True:
        await asyncio.sleep(KEEPALIVE_SECONDS)
        await dg.send(json.dumps({"type": "KeepAlive"}))


@app.websocket("/ws/transcribe")
async def transcribe(browser: WebSocket):
    await browser.accept()

    if not DEEPGRAM_API_KEY:
        await browser.send_json({"type": "error", "message": "DEEPGRAM_API_KEY is not set on the server."})
        await browser.close()
        return

    headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}
    try:
        async with websockets.connect(deepgram_url(), additional_headers=headers) as dg:
            await browser.send_json({"type": "status", "message": "listening"})

            audio_task = asyncio.create_task(forward_audio(browser, dg))
            transcript_task = asyncio.create_task(forward_transcripts(dg, browser))
            keep_task = asyncio.create_task(keep_alive(dg))
            try:
                done, _ = await asyncio.wait(
                    {audio_task, transcript_task}, return_when=asyncio.FIRST_COMPLETED
                )
                if audio_task in done:
                    # Browser stopped: give Deepgram a moment to send the final results.
                    with suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(transcript_task, timeout=FLUSH_TIMEOUT_SECONDS)
            finally:
                for task in (audio_task, transcript_task, keep_task):
                    task.cancel()
                await asyncio.gather(audio_task, transcript_task, keep_task, return_exceptions=True)
    except Exception as exc:
        with suppress(Exception):
            await browser.send_json({"type": "error", "message": f"Deepgram error: {exc}"})
    finally:
        with suppress(Exception):
            await browser.close()
