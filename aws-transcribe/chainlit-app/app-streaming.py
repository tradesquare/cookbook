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

# แก้ไขการจัดการ AWS credentials
import os
import boto3

# ลบบรรทัดนี้ออกสำหรับ ECS deployment
# boto3.setup_default_session(profile_name=AWS_PROFILE)

# ใช้ environment variables แทน profile
aws_region = os.getenv("AWS_REGION", "us-east-1")

# สร้าง client โดยไม่ระบุ profile
bedrock_client = boto3.client('bedrock-runtime', region_name=aws_region)

# แก้ไข environment variable validation
BOTNOI_API_KEY = os.getenv("BOTNOI_API_KEY")
if not BOTNOI_API_KEY:
    raise ValueError("BOTNOI_API_KEY must be set")

# เพิ่ม health check endpoint (optional)
@cl.on_chat_start
async def start():
    cl.user_session.set("message_history", [])
    await cl.Message(
        content="Welcome to Chainlit x AWS example! Press `p` to talk!",
    ).send()

# หรือเพิ่ม HTTP endpoint สำหรับ health check

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
    500  # Adjust based on your audio level (e.g., lower for quieter audio)
)
SILENCE_TIMEOUT = 3000.0  # Milliseconds of silence to consider the turn finished

@cl.step(type="tool")
async def speech_to_text(audio_buffer):
    try:
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
    except Exception as e:
        logging.error(f"Speech to text error: {str(e)}")
        return "Sorry, I couldn't process your audio."


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
    
    # Try different Claude models in order of preference
    model_ids = [
        "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "anthropic.claude-3-sonnet-20240229-v1:0",
        # Add more fallback models if needed
    ]
    
    assistant_message = None
    for model_id in model_ids:
        try:
            # Call Bedrock
            response = bedrock_client.invoke_model(
                modelId=model_id,
                body=json.dumps(body)
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            assistant_message = response_body['content'][0]['text']
            break  # Success, exit the loop
            
        except Exception as e:
            logging.warning(f"Failed to use model {model_id}: {str(e)}")
            continue  # Try next model
    
    if not assistant_message:
        # If all models fail, provide a fallback response
        assistant_message = "I'm sorry, I'm currently unable to process your request. Please try again later."
        logging.error("All Bedrock models failed to respond")
    
    # Add assistant response to history
    message_history.append({"role": "assistant", "content": assistant_message})
    
    return assistant_message


@cl.on_audio_start
async def on_audio_start():
    cl.user_session.set("silent_duration_ms", 0)
    cl.user_session.set("is_speaking", False) # Initialize speaking state
    cl.user_session.set("audio_chunks", []) # Initialize audio chunks Comment= None or []?
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
        return

    #audio_chunks = cl.user_session.get("audio_chunks")
    last_elapsed_time = cl.user_session.get("last_elapsed_time")
    silent_duration_ms = cl.user_session.get("silent_duration_ms")
    is_speaking = cl.user_session.get("is_speaking")

    # Calculate the time difference between this chunk and the previous one
    time_diff_ms = chunk.elapsedTime - last_elapsed_time
    cl.user_session.set("last_elapsed_time", chunk.elapsedTime)

    # Compute the RMS (root mean square) energy of the audio chunk
    audio_chunk = np.frombuffer(chunk.data, dtype=np.int16)
    audio_energy = np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2))

    if audio_energy < SILENCE_THRESHOLD:
        # Audio is considered silent
        silent_duration_ms += time_diff_ms
        cl.user_session.set("silent_duration_ms", silent_duration_ms)
        if silent_duration_ms >= SILENCE_TIMEOUT and is_speaking:
            cl.user_session.set("is_speaking", False)
            await process_audio()
    else:
        # Audio is not silent, reset silence timer and mark as speaking
        cl.user_session.set("silent_duration_ms", 0)
        if not is_speaking:
            cl.user_session.set("is_speaking", True)


async def process_audio():
    # Get the audio buffer from the session
    if audio_chunks := cl.user_session.get("audio_chunks"):
        
        # Log the number of audio chunks for debugging
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

        # Reset buffer position to read the WAV file
        wav_buffer.seek(0)

        # Check audio duration before processing
        with wave.open(wav_buffer, 'rb') as wav_file_check:
            frames = wav_file_check.getnframes()
            rate = wav_file_check.getframerate()
            duration = frames / float(rate)
            
        if duration <= 0.5:
            print("The audio is too short, please try again.")
            cl.user_session.set("audio_chunks", [])
            return

        # Reset buffer position again for transcription
        wav_buffer.seek(0)
        audio_buffer = wav_buffer.getvalue()

        # Clear audio chunks from session
        cl.user_session.set("audio_chunks", [])

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


@cl.on_audio_end
async def on_audio_end():
    """จัดการเมื่อผู้ใช้กดปุ่ม 'p' เพื่อหยุดการบันทึกเสียง"""
    # ประมวลผลเสียงที่บันทึกได้
    await process_audio()


@cl.on_message
async def on_message(message: cl.Message):
    await cl.Message(content="This is a voice demo, press P to start!").send()

