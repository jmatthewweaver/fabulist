"""
Voice I/O: Whisper transcription (input) + OpenAI TTS (output).
Phase 4 feature — these endpoints are stubs until the voice layer is wired into the UI.
"""
from openai import OpenAI

from ..config import settings

_client = OpenAI(api_key=settings.openai_api_key)


async def transcribe(audio_bytes: bytes, filename: str = "audio.webm") -> str:
    """Transcribe audio bytes to text via Whisper."""
    import io
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = filename
    transcript = _client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="text",
    )
    return transcript.strip()


def synthesize_stream(text: str):
    """Stream TTS audio bytes. Yields chunks for streaming response."""
    with _client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="nova",
        input=text,
        response_format="mp3",
    ) as response:
        yield from response.iter_bytes(chunk_size=4096)
