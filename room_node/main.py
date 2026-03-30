"""
Room node entry point.

Starts the audio capture loop, runs inference on each utterance, and
dispatches payloads to the pipeline worker.  Designed to run as a systemd
service on the Raspberry Pi 5.

Graceful shutdown is handled via SIGTERM (sent by systemd on stop).
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys

from room_node.capture import AudioCapture
from room_node.config import RoomNodeConfig
from room_node.doa import DOAReader
from room_node.hailo_inference import InferenceEngine
from room_node.sender import PayloadSender


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )


async def run(config: RoomNodeConfig) -> None:
    """Main async loop: capture → infer → send."""
    logger = logging.getLogger(__name__)
    logger.info("Room node starting — room=%s", config.room_name)

    doa_reader = DOAReader()
    engine = InferenceEngine(
        hailo_enabled=config.hailo_enabled,
        whisper_hef=config.hailo_whisper_hef,
        emotion_hef=config.hailo_emotion_hef,
        whisper_fallback_model=config.whisper_fallback_model,
    )
    sender = PayloadSender(
        blackmagic_url=config.blackmagic_url,
        room_name=config.room_name,
        whisper_confidence_threshold=config.whisper_confidence_threshold,
        max_retries=config.http_max_retries,
        retry_backoff_s=config.http_retry_backoff_s,
        sample_rate=config.sample_rate,
        queue_maxsize=config.offline_queue_maxsize,
    )
    capture = AudioCapture(
        device_index=config.device_index,
        sample_rate=config.sample_rate,
        vad_threshold=config.vad_threshold,
        vad_min_silence_ms=config.vad_min_silence_ms,
        vad_speech_pad_ms=config.vad_speech_pad_ms,
        max_utterance_s=config.max_utterance_s,
    )

    # Run the blocking capture loop in a thread so asyncio stays live
    loop = asyncio.get_event_loop()
    stop_event = asyncio.Event()

    def _handle_sigterm(*_):
        logger.info("SIGTERM received — shutting down")
        loop.call_soon_threadsafe(stop_event.set)

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    async def _capture_and_process():
        for utterance in capture.iter_utterances():
            if stop_event.is_set():
                break

            doa = doa_reader.read()
            result = engine.run(utterance)

            if not result.transcript:
                logger.debug("Empty transcript — skipping utterance")
                continue

            await sender.send(
                audio=utterance,
                transcript=result.transcript,
                whisper_confidence=result.whisper_confidence,
                whisper_model=result.whisper_model,
                doa=doa,
                emotion_valence=result.emotion_valence,
                emotion_arousal=result.emotion_arousal,
                voiceprint=result.voiceprint,
            )

    capture_task = asyncio.create_task(_capture_and_process())
    await stop_event.wait()
    capture.stop()
    capture_task.cancel()
    logger.info("Room node shut down cleanly")


def main() -> None:
    config = RoomNodeConfig()
    _configure_logging(config.log_level)
    try:
        asyncio.run(run(config))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
