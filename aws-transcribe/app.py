import os
import io
import wave
import boto3
import numpy as np
import audioop
import json

import chainlit as cl

AWS_PROFILE = os.getenv("PROFILE")

session = boto3.Session(profile_name=AWS_PROFILE)
transcribe_client = session.client('transcribe')
polly_client = session.client('polly')
bedrock_client = session.client('bedrock-runtime')

if not AWS_PROFILE:
    raise ValueError(
        "PROFILE must be set"
    )


# Define a threshold for detecting silence and a timeout for ending a turn
SILENCE_THRESHOLD = (
    3500  # Adjust based on your audio level (e.g., lower for quieter audio)
)
SILENCE_TIMEOUT = 1300.0  # Seconds of silence to consider the turn finished


@cl.step(type="tool")
async def speech_to_text(audio_file):
    # For now, using a placeholder since Bedrock doesn't have direct speech-to-text
    # You would need to use Amazon Transcribe service for this functionality
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_file.write(audio_file[1])
        temp_file.flush()
        
        # This is a simplified placeholder - you'd need to implement Transcribe job
        return "[Speech transcription would go here - implement with Amazon Transcribe]"


@cl.step(type="tool")
async def text_to_speech(text: str, mime_type: str):
    response = polly_client.synthesize_speech(
        Text=text,
        OutputFormat='mp3',
        VoiceId='Joanna'
    )
    
    buffer = io.BytesIO()
    buffer.name = "output_audio.mp3"
    buffer.write(response['AudioStream'].read())
    buffer.seek(0)
    
    return buffer.name, buffer.read()


@cl.step(type="tool")
async def generate_text_answer(transcription):
    message_history = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": transcription})

    # Convert message history to Claude format
    messages = []
    for msg in message_history:
        if msg["role"] == "user":
            messages.append({"role": "user", "content": msg["content"]})
        elif msg["role"] == "assistant":
            messages.append({"role": "assistant", "content": msg["content"]})

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "messages": messages
    })

    response = bedrock_client.invoke_model(
        body=body,
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        accept="application/json",
        contentType="application/json"
    )

    response_body = json.loads(response.get('body').read())
    answer = response_body['content'][0]['text']
    
    message_history.append({"role": "assistant", "content": answer})
    return answer


@cl.on_chat_start
async def start():
    cl.user_session.set("message_history", [])
    await cl.Message(
        content="Welcome to Chainlit x Whisper example! Press `p` to talk!",
    ).send()


@cl.on_audio_start
async def on_audio_start():
    cl.user_session.set("silent_duration_ms", 0)
    cl.user_session.set("is_speaking", False)
    cl.user_session.set("audio_chunks", [])
    return True


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk):
    audio_chunks = cl.user_session.get("audio_chunks")

    if audio_chunks is not None:
        audio_chunk = np.frombuffer(chunk.data, dtype=np.int16)
        audio_chunks.append(audio_chunk)

    # If this is the first chunk, initialize timers and state
    if chunk.isStart:
        cl.user_session.set("last_elapsed_time", chunk.elapsedTime)
        cl.user_session.set("is_speaking", True)
        return

    audio_chunks = cl.user_session.get("audio_chunks")
    last_elapsed_time = cl.user_session.get("last_elapsed_time")
    silent_duration_ms = cl.user_session.get("silent_duration_ms")
    is_speaking = cl.user_session.get("is_speaking")

    # Calculate the time difference between this chunk and the previous one
    time_diff_ms = chunk.elapsedTime - last_elapsed_time
    cl.user_session.set("last_elapsed_time", chunk.elapsedTime)

    # Compute the RMS (root mean square) energy of the audio chunk
    audio_energy = audioop.rms(
        chunk.data, 2
    )  # Assumes 16-bit audio (2 bytes per sample)

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

        cl.user_session.set("audio_chunks", [])

    frames = wav_file.getnframes()
    rate = wav_file.getframerate()

    duration = frames / float(rate)
    if duration <= 1.71:
        print("The audio is too short, please try again.")
        return

    audio_buffer = wav_buffer.getvalue()

    input_audio_el = cl.Audio(content=audio_buffer, mime="audio/wav")

    whisper_input = ("audio.wav", audio_buffer, "audio/wav")
    transcription = await speech_to_text(whisper_input)

    await cl.Message(
        author="You",
        type="user_message",
        content=transcription,
        elements=[input_audio_el],
    ).send()

    answer = await generate_text_answer(transcription)

    output_name, output_audio = await text_to_speech(answer, "audio/mp3")

    output_audio_el = cl.Audio(
        auto_play=True,
        mime="audio/mp3",
        content=output_audio,
    )

    await cl.Message(content=answer, elements=[output_audio_el]).send()


@cl.on_message
async def on_message(message: cl.Message):
    await cl.Message(content="This is a voice demo, press P to start!").send()
