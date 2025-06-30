import os
import io
import wave
import boto3
import numpy as np
import audioop
import json
import requests
import asyncio

import logging
import time
import uuid

from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent

import chainlit as cl

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
# Lowered threshold for better sensitivity with Thai speech
SILENCE_THRESHOLD = 1500  # Lowered from 3500 for better Thai speech detection
SILENCE_TIMEOUT = 3000.0  # Increased from 2000ms to 3000ms for Thai speech patterns

# Language-specific settings
THAI_SILENCE_THRESHOLD = 1200  # Even lower threshold for Thai
THAI_SILENCE_TIMEOUT = 3500.0  # Longer timeout for Thai speech patterns
ENGLISH_SILENCE_THRESHOLD = 2000
ENGLISH_SILENCE_TIMEOUT = 2500.0

# Manual language override for testing - set this to force a specific language
FORCE_LANGUAGE = "thai"  # Set to "thai" or "english" to override detection, or None for auto-detection

@cl.step(type="tool")
async def speech_to_text(audio_buffer):
    #
    # Set up streaming client
    client = TranscribeStreamingClient(region=aws_region)
    
    # Start stream transcription
    stream = await client.start_stream_transcription(
        language_code="th-TH",
        #language_code="en-US",
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
    
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    
    #get audio url in response
    # get audio_url in response.content
    audio_url = json.loads(response.content.decode('utf-8'))['audio_url']
    
    return "output_audio.mp3", response.content, audio_url


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
    
    # Call Bedrock
    response = bedrock_client.invoke_model(
        #modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        #modelId="anthropic.claude-3-7-sonnet-20250219-v1:0",
        modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
        body=json.dumps(body)
    )
    
    # Parse response
    response_body = json.loads(response['body'].read())
    assistant_message = response_body['content'][0]['text']
    
    # Add assistant response to history
    message_history.append({"role": "assistant", "content": assistant_message})
    
    return assistant_message


@cl.on_chat_start
async def start():
    cl.user_session.set("message_history", [])
    await cl.Message(
        content="Welcome to Chainlit x AWS example! Press `p` to talk!",
    ).send()


@cl.on_audio_start
async def on_audio_start():
    cl.user_session.set("silent_duration_ms", 0)
    cl.user_session.set("is_speaking", False) # Initialize speaking state
    cl.user_session.set("audio_chunks", []) # Initialize audio chunks
    cl.user_session.set("energy_history", []) # Initialize energy history for moving average
    cl.user_session.set("speech_start_time", 0) # Initialize speech start time
    return True


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk):
    audio_chunks = cl.user_session.get("audio_chunks")

    # Initialize audio_chunks if it doesn't exist
    if audio_chunks is not None:
        audio_chunk = np.frombuffer(chunk.data, dtype=np.int16)
        audio_chunks.append(audio_chunk)

    # If this is the first chunk, initialize timers and state
    if chunk.isStart:
        cl.user_session.set("last_elapsed_time", chunk.elapsedTime)
        cl.user_session.set("is_speaking", True)
        cl.user_session.set("speech_start_time", chunk.elapsedTime)
        return

    last_elapsed_time = cl.user_session.get("last_elapsed_time")
    silent_duration_ms = cl.user_session.get("silent_duration_ms")
    is_speaking = cl.user_session.get("is_speaking")
    speech_start_time = cl.user_session.get("speech_start_time", chunk.elapsedTime)

    # Calculate the time difference between this chunk and the previous one
    time_diff_ms = chunk.elapsedTime - last_elapsed_time
    cl.user_session.set("last_elapsed_time", chunk.elapsedTime)

    # Compute the RMS (root mean square) energy of the audio chunk
    audio_chunk = np.frombuffer(chunk.data, dtype=np.int16)
    audio_energy = np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2))

    # Detect language and get appropriate parameters
    detected_language = detect_language_from_audio_energy(audio_chunks)
    silence_threshold, silence_timeout = get_silence_parameters(detected_language)
    
    # Debug logging every 50 chunks to avoid spam
    chunk_count = cl.user_session.get("chunk_count", 0)
    chunk_count += 1
    cl.user_session.set("chunk_count", chunk_count)
    
    if chunk_count % 50 == 0:
        logging.info(f"Language: {detected_language}, Energy: {audio_energy:.2f}, "
                    f"Threshold: {silence_threshold}, Timeout: {silence_timeout}ms")
    
    # Minimum speech duration before allowing cutoff (prevent very short clips)
    min_speech_duration = 1500  # 1.5 seconds minimum
    total_speech_duration = chunk.elapsedTime - speech_start_time

    # Enhanced silence detection with multiple criteria
    is_silent = audio_energy < silence_threshold
    
    # Additional check: use a moving average for more stable detection
    if not hasattr(cl.user_session, 'energy_history'):
        cl.user_session.set("energy_history", [])
    
    energy_history = cl.user_session.get("energy_history")
    energy_history.append(audio_energy)
    
    # Keep only last 5 readings for moving average
    if len(energy_history) > 5:
        energy_history.pop(0)
    
    avg_energy = np.mean(energy_history)
    is_consistently_silent = avg_energy < silence_threshold

    if is_consistently_silent:
        # Audio is considered silent
        silent_duration_ms += time_diff_ms
        cl.user_session.set("silent_duration_ms", silent_duration_ms)
        
        # Only process if we have minimum speech duration AND silence timeout
        if (silent_duration_ms >= silence_timeout and 
            is_speaking and 
            total_speech_duration >= min_speech_duration):
            cl.user_session.set("is_speaking", False)
            await process_audio()
    else:
        # Audio is not silent, reset silence timer and mark as speaking
        cl.user_session.set("silent_duration_ms", 0)
        if not is_speaking:
            cl.user_session.set("is_speaking", True)
            cl.user_session.set("speech_start_time", chunk.elapsedTime)


async def process_audio():
    # Get the audio buffer from the session
    if audio_chunks := cl.user_session.get("audio_chunks"):
        
        #logging of the list of audio chunks for debugging using logging
        logging.info(f"Audio chunks received------------------: {len(audio_chunks)}")
        logging.info(f"Audio chunks received------------------: {len(audio_chunks)}")
        logging.info(f"Audio chunks received------------------: {len(audio_chunks)}")
        logging.info(f"Audio chunks received------------------: {len(audio_chunks)}")
        logging.info(f"Audio chunks received------------------: {len(audio_chunks)}")
    
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

        # Open the WAV file to check its properties
        cl.user_session.set("audio_chunks", [])

    frames = wav_file.getnframes()
    rate = wav_file.getframerate()

    duration = frames / float(rate)
    if duration <= 0.5:
        print("The audio is too short, please try again.")
        return

    audio_buffer = wav_buffer.getvalue()

    input_audio_el = cl.Audio(content=audio_buffer, mime="audio/wav")

    transcription = await speech_to_text(audio_buffer)

    await cl.Message(
        author="You",
        type="user_message",
        content=transcription,
        elements=[input_audio_el],
    ).send()

    answer = await generate_text_answer(transcription)

    output_name, output_audio, audio_url = await text_to_speech(answer, "audio/mp3")

    output_audio_el = cl.Audio(
        #get url from botnoi api response
        url=audio_url,
        auto_play=True,
        mime="audio/mp3",
        content=output_audio,
    )

    await cl.Message(content=answer, elements=[output_audio_el]).send()


@cl.on_message
async def on_message(message: cl.Message):
    await cl.Message(content="This is a voice demo, press P to start!").send()

def detect_language_from_audio_energy(audio_chunks, recent_chunks_count=10):
    """
    Simple heuristic to detect language based on recent audio patterns.
    Thai speech often has different energy patterns than English.
    """
    if not audio_chunks or len(audio_chunks) < recent_chunks_count:
        return "unknown"
    
    # Analyze recent chunks for energy variance
    recent_chunks = audio_chunks[-recent_chunks_count:]
    energies = []
    
    for chunk in recent_chunks:
        if isinstance(chunk, np.ndarray):
            energy = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))
            energies.append(energy)
    
    if not energies:
        return "unknown"
    
    # Thai speech often has more varied energy patterns
    energy_variance = np.var(energies)
    mean_energy = np.mean(energies)
    
    # Heuristic: Thai speech tends to have higher variance in energy
    if energy_variance > mean_energy * 0.3:
        return "thai"
    else:
        return "english"

def get_silence_parameters(detected_language="unknown"):
    """
    Get appropriate silence detection parameters based on detected language.
    """
    # Check for manual override first
    if FORCE_LANGUAGE:
        detected_language = FORCE_LANGUAGE
        
    if detected_language == "thai":
        return THAI_SILENCE_THRESHOLD, THAI_SILENCE_TIMEOUT
    elif detected_language == "english":
        return ENGLISH_SILENCE_THRESHOLD, ENGLISH_SILENCE_TIMEOUT
    else:
        # Default to more permissive settings (Thai parameters)
        return THAI_SILENCE_THRESHOLD, THAI_SILENCE_TIMEOUT


