import os
import io
import wave
import boto3
import numpy as np
import audioop
import json
import requests
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import logging
import time
import uuid

from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent

import chainlit as cl

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

AWS_PROFILE = os.getenv("PROFILE")
BOTNOI_API_KEY = os.getenv("BOTNOI_API_KEY")

boto3.setup_default_session(profile_name=AWS_PROFILE)

session = boto3.Session(profile_name=AWS_PROFILE)
bedrock_client = session.client('bedrock-runtime')

# Get AWS region from session
aws_region = session.region_name or 'us-east-1'

if not AWS_PROFILE or not BOTNOI_API_KEY:
    raise ValueError(
        "PROFILE and BOTNOI_API_KEY must be set"
    )

class TranscribeEventHandler(TranscriptResultStreamHandler):
    def __init__(self, output_stream):
        super().__init__(output_stream)
        self.transcript_text = ""
        
    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        results = transcript_event.transcript.results
        for result in results:
            if not result.is_partial:
                for alt in result.alternatives:
                    self.transcript_text += alt.transcript + " "


# Define a threshold for detecting silence and a timeout for ending a turn
SILENCE_THRESHOLD = (
    3500  # Adjust based on your audio level (e.g., lower for quieter audio)
)
SILENCE_TIMEOUT = 2000.0  # Milliseconds of silence to consider the turn finished
INTERRUPTION_THRESHOLD = 4000  # Higher threshold for detecting interruption during bot speech

# Global event for handling interruptions
interruption_event = asyncio.Event()
bot_speaking_event = asyncio.Event()
executor = ThreadPoolExecutor(max_workers=4)

@cl.step(type="tool")
async def speech_to_text(audio_buffer):
    #
    # Set up streaming client
    client = TranscribeStreamingClient(region=aws_region)
    
    # Start stream transcription
    stream = await client.start_stream_transcription(
        #language_code="th-TH",
        language_code="en-US",
        #identify_multiple_languages=True,
        #language_options=["th-TH", "en-US"],
        media_sample_rate_hz=24000,
        media_encoding="pcm",
    )
    
    # Create event handler
    handler = TranscribeEventHandler(stream.output_stream)
    
    async def send_audio():
        # Convert WAV buffer to PCM chunks
        wav_io = io.BytesIO(audio_buffer)
        with wave.open(wav_io, 'rb') as wav_file:
            chunk_size = 1024 * 2  # 2KB chunks
            while True:
                chunk = wav_file.readframes(chunk_size)
                if not chunk:
                    break
                await stream.input_stream.send_audio_event(audio_chunk=chunk)
        await stream.input_stream.end_stream()
    
    # Process audio and handle events concurrently
    await asyncio.gather(send_audio(), handler.handle_events())
    
    return handler.transcript_text.strip()


@cl.step(type="tool")
async def text_to_speech(text: str, mime_type: str):
    """Legacy text_to_speech function - redirects to interruptible version"""
    return await interruptible_text_to_speech(text, mime_type)


@cl.step(type="tool")
async def generate_text_answer(transcription):
    message_history = cl.user_session.get("message_history")
    
    # Add user message to history
    message_history.append({"role": "user", "content": transcription})
    
    # Convert message history to Claude format
    claude_messages = []
    for msg in message_history:
        if msg["role"] == "user":
            claude_messages.append({"role": "user", "content": msg["content"]})
        elif msg["role"] == "assistant":
            claude_messages.append({"role": "assistant", "content": msg["content"]})
    
    # Prepare request for Claude
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1500,
        "temperature": 0.2,
        "messages": claude_messages
    }
    
    # Call Bedrock with interruption check
    try:
        response = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: bedrock_client.invoke_model(
                    modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
                    body=json.dumps(body)
                )
            ),
            timeout=10.0  # 10 second timeout
        )
        
        # Check for interruption after API call
        if interruption_event.is_set():
            logging.info("Response generation interrupted by user speech")
            return None
            
    except asyncio.TimeoutError:
        logging.warning("Claude API call timed out")
        return "I'm having trouble processing your request right now. Please try again."
    
    # Parse response
    response_body = json.loads(response['body'].read())
    assistant_message = response_body['content'][0]['text']
    
    # Add assistant response to history only if not interrupted
    if not interruption_event.is_set():
        message_history.append({"role": "assistant", "content": assistant_message})
    
    return assistant_message


@cl.step(type="tool")
async def interruptible_text_to_speech(text: str, mime_type: str):
    """Text-to-speech with interruption capability"""
    try:
        # Check for interruption before starting TTS
        if interruption_event.is_set():
            return None, None, None
            
        # Detect if text contains Thai characters
        has_thai = any('\u0e00' <= char <= '\u0e7f' for char in text)
        
        url = "https://api-voice.botnoi.ai/openapi/v1/generate_audio"
        payload = {
            "text": text,
            "speaker": "1",
            "volume": "1",
            "speed": 1,
            "type_media": "mp3",
            "save_file": "true",
            "language": "th" if has_thai else "en",
        }
        headers = {
            'Botnoi-Token': BOTNOI_API_KEY,
            'Content-Type': 'application/json'
        }
        
        # Make API call with interruption check
        response = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: requests.post(url, headers=headers, json=payload)
            ),
            timeout=15.0
        )
        
        # Check for interruption after API call
        if interruption_event.is_set():
            logging.info("TTS generation interrupted by user speech")
            return None, None, None
            
        response.raise_for_status()
        
        # Get audio url in response
        audio_url = json.loads(response.content.decode('utf-8'))['audio_url']
        
        return "output_audio.mp3", response.content, audio_url
        
    except asyncio.TimeoutError:
        logging.warning("TTS API call timed out")
        return None, None, None
    except Exception as e:
        logging.error(f"TTS error: {e}")
        return None, None, None


@cl.on_chat_start
async def start():
    cl.user_session.set("message_history", [])
    cl.user_session.set("bot_is_responding", False)
    # Reset global events
    interruption_event.clear()
    bot_speaking_event.clear()
    await cl.Message(
        content="Welcome to Chainlit x AWS example! Press `p` to talk! You can interrupt me anytime while I'm speaking.",
    ).send()


@cl.on_audio_start
async def on_audio_start():
    cl.user_session.set("silent_duration_ms", 0)
    cl.user_session.set("is_speaking", False)
    cl.user_session.set("audio_chunks", [])
    
    # Reset interruption event when user starts speaking
    interruption_event.clear()
    return True


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk):
    audio_chunks = cl.user_session.get("audio_chunks")
    bot_is_responding = cl.user_session.get("bot_is_responding", False)

    # Initialize audio_chunks if it doesn't exist
    if audio_chunks is not None:
        audio_chunk = np.frombuffer(chunk.data, dtype=np.int16)
        audio_chunks.append(audio_chunk)

    # If this is the first chunk, initialize timers and state
    if chunk.isStart:
        cl.user_session.set("last_elapsed_time", chunk.elapsedTime)
        cl.user_session.set("is_speaking", True)
        return

    last_elapsed_time = cl.user_session.get("last_elapsed_time")
    silent_duration_ms = cl.user_session.get("silent_duration_ms")
    is_speaking = cl.user_session.get("is_speaking")

    # Calculate the time difference between this chunk and the previous one
    time_diff_ms = chunk.elapsedTime - last_elapsed_time
    cl.user_session.set("last_elapsed_time", chunk.elapsedTime)

    # Compute the RMS (root mean square) energy of the audio chunk
    audio_chunk = np.frombuffer(chunk.data, dtype=np.int16)
    audio_energy = np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2))

    # Check for interruption if bot is speaking
    if bot_is_responding and audio_energy > INTERRUPTION_THRESHOLD:
        logging.info(f"User interruption detected! Energy: {audio_energy}")
        interruption_event.set()
        bot_speaking_event.clear()
        cl.user_session.set("bot_is_responding", False)
        # Continue processing the interrupting speech
        cl.user_session.set("is_speaking", True)
        cl.user_session.set("silent_duration_ms", 0)
        return

    # Use different thresholds based on bot state
    silence_threshold = SILENCE_THRESHOLD if not bot_is_responding else INTERRUPTION_THRESHOLD

    if audio_energy < silence_threshold:
        # Audio is considered silent
        silent_duration_ms += time_diff_ms
        cl.user_session.set("silent_duration_ms", silent_duration_ms)
        if silent_duration_ms >= SILENCE_TIMEOUT and is_speaking:
            cl.user_session.set("is_speaking", False)
            if not bot_is_responding:  # Only process if bot is not responding
                await process_audio()
    else:
        # Audio is not silent, reset silence timer and mark as speaking
        cl.user_session.set("silent_duration_ms", 0)
        if not is_speaking:
            cl.user_session.set("is_speaking", True)


async def process_audio():
    # Get the audio buffer from the session
    if audio_chunks := cl.user_session.get("audio_chunks"):
        
        logging.info(f"Audio chunks received: {len(audio_chunks)}")
    
        # Concatenate all chunks
        concatenated = np.concatenate(list(audio_chunks))

        # Create an in-memory binary stream
        wav_buffer = io.BytesIO()

        # Create WAV file with proper parameters
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)  # mono
            wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
            wav_file.setframerate(24000)  # sample rate (24kHz PCM)
            wav_file.writeframes(concatenated.tobytes())

        # Reset buffer position
        wav_buffer.seek(0)

        # Check audio duration
        with wave.open(wav_buffer, 'rb') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            duration = frames / float(rate)
            
        if duration <= 0.5:
            print("The audio is too short, please try again.")
            cl.user_session.set("audio_chunks", [])
            return

        # Reset buffer position again
        wav_buffer.seek(0)
        audio_buffer = wav_buffer.getvalue()
        
        # Clear audio chunks
        cl.user_session.set("audio_chunks", [])

        input_audio_el = cl.Audio(content=audio_buffer, mime="audio/wav")

        # Transcribe the audio
        transcription = await speech_to_text(audio_buffer)

        await cl.Message(
            author="You",
            type="user_message",
            content=transcription,
            elements=[input_audio_el],
        ).send()

        # Set bot responding state
        cl.user_session.set("bot_is_responding", True)
        bot_speaking_event.set()
        
        # Generate response with interruption handling
        answer = await generate_text_answer(transcription)
        
        # Check if response was interrupted
        if answer is None or interruption_event.is_set():
            logging.info("Response interrupted, not proceeding with TTS")
            cl.user_session.set("bot_is_responding", False)
            bot_speaking_event.clear()
            return

        # Generate TTS with interruption handling
        output_name, output_audio, audio_url = await interruptible_text_to_speech(answer, "audio/mp3")
        
        # Check if TTS was interrupted
        if output_name is None or interruption_event.is_set():
            logging.info("TTS interrupted, not sending audio response")
            cl.user_session.set("bot_is_responding", False)
            bot_speaking_event.clear()
            return

        # Send response only if not interrupted
        if not interruption_event.is_set():
            output_audio_el = cl.Audio(
                url=audio_url,
                auto_play=True,
                mime="audio/mp3",
                content=output_audio,
            )

            await cl.Message(content=answer, elements=[output_audio_el]).send()
            
            # Monitor for interruptions during audio playback
            asyncio.create_task(monitor_audio_playback())
        
        # Reset bot responding state
        cl.user_session.set("bot_is_responding", False)
        bot_speaking_event.clear()


async def monitor_audio_playback():
    """Monitor for interruptions during audio playback"""
    try:
        # Wait for interruption or timeout (assume max 30 seconds for audio)
        await asyncio.wait_for(interruption_event.wait(), timeout=30.0)
        logging.info("Audio playback interrupted by user speech")
    except asyncio.TimeoutError:
        # Audio finished playing normally
        pass
    finally:
        # Ensure bot responding state is cleared
        cl.user_session.set("bot_is_responding", False)
        bot_speaking_event.clear()


@cl.on_chat_end
async def on_chat_end():
    """Clean up resources when chat session ends"""
    global interruption_event, bot_speaking_event
    
    # Clear all events
    interruption_event.clear()
    bot_speaking_event.clear()
    
    # Reset session state
    cl.user_session.set("bot_is_responding", False)
    cl.user_session.set("is_speaking", False)
    cl.user_session.set("audio_chunks", [])
    
    logging.info("Chat session ended, resources cleaned up")


async def reset_interruption_state():
    """Reset interruption state for new conversation turn"""
    global interruption_event
    interruption_event.clear()
    cl.user_session.set("bot_is_responding", False)


@cl.on_message
async def on_message(message: cl.Message):
    await cl.Message(content="This is a voice demo, press P to start! You can interrupt me anytime while I'm speaking.").send()


