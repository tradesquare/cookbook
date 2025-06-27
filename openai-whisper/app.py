import os
import io
import wave
import numpy as np
import logging
import boto3
import json
import uuid
import time
import chainlit as cl
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# AWS configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_PROFILE = os.getenv("AWS_PROFILE", "")

# Knowledge Base Configuration
KNOWLEDGE_BASE_ENABLED = os.getenv("KNOWLEDGE_BASE_ENABLED", "false").lower() == "true"
logger.info(f"Knowledge Base enabled: {KNOWLEDGE_BASE_ENABLED}")
KNOWLEDGE_BASE_ID = os.getenv("KNOWLEDGE_BASE_ID", "")
KNOWLEDGE_BASE_MODEL_ARN = os.getenv("KNOWLEDGE_BASE_MODEL_ARN", "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0")

# Initialize AWS clients
try:
    if AWS_PROFILE:
        session = boto3.Session(profile_name=AWS_PROFILE)
        logger.info(f"Using AWS profile: {AWS_PROFILE}")
        transcribe_client = session.client('transcribe', region_name=AWS_REGION)
        bedrock_client = session.client('bedrock-runtime', region_name=AWS_REGION)
        bedrock_agent_client = session.client('bedrock-agent-runtime', region_name=AWS_REGION) if KNOWLEDGE_BASE_ENABLED else None
        s3_client = session.client('s3', region_name=AWS_REGION)
        polly_client = session.client('polly', region_name=AWS_REGION)
    else:
        logger.info("Using default AWS session (IAM roles/environment variables)")
        transcribe_client = boto3.client('transcribe', region_name=AWS_REGION)
        bedrock_client = boto3.client('bedrock-runtime', region_name=AWS_REGION)
        bedrock_agent_client = boto3.client('bedrock-agent-runtime', region_name=AWS_REGION) if KNOWLEDGE_BASE_ENABLED else None
        s3_client = boto3.client('s3', region_name=AWS_REGION)
        polly_client = boto3.client('polly', region_name=AWS_REGION)
    logger.info(f"AWS clients initialized successfully for region: {AWS_REGION}")
except Exception as e:
    logger.error(f"Failed to initialize AWS clients: {e}")
    raise

# S3 bucket for temporary audio files
S3_BUCKET = os.getenv("S3_BUCKET", "chainlit-voice-chat-audio-dev-654654383273")

def test_s3_connectivity():
    """Test S3 connectivity and bucket access"""
    try:
        # HEAD request checks bucket existence without downloading data
        # This is a lightweight operation that verifies:
        # 1. Bucket exists
        # 2. We have permission to access it
        # 3. AWS credentials are valid
        s3_client.head_bucket(Bucket=S3_BUCKET)
        logger.info(f"S3 bucket '{S3_BUCKET}' is accessible")
        return True
    except Exception as e:
        # Common failures: bucket doesn't exist, no permissions, invalid credentials
        logger.error(f"S3 connectivity test failed: {e}")
        logger.error(f"Please ensure the bucket '{S3_BUCKET}' exists and AWS credentials are properly configured")
        return False

# Test S3 connectivity at startup
if not test_s3_connectivity():
    logger.warning("S3 connectivity test failed - the application may not work properly")

# FastAPI app for health checks
app = FastAPI()

@app.get("/health")
async def health_check():
    """Health check endpoint for ECS load balancer"""
    try:
        # Create base health status with application metadata
        # Load balancers use this to determine if instance is healthy
        health_status = {
            "status": "healthy",  # Default to healthy, will change if issues found
            "timestamp": time.time(),  # Unix timestamp for when check was performed
            "aws_region": AWS_REGION,  # Which AWS region we're operating in
            "s3_bucket": S3_BUCKET  # Which S3 bucket we depend on
        }
        
        # Test critical dependency: S3 bucket access
        # If S3 fails, app can't store audio files for transcription
        try:
            s3_client.head_bucket(Bucket=S3_BUCKET)
            health_status["s3_status"] = "accessible"
        except Exception as s3_error:
            # S3 failure means degraded service (not completely down)
            health_status["s3_status"] = f"error: {str(s3_error)}"
            health_status["status"] = "degraded"  # Still serving but with issues
        
        # Return 200 OK even if degraded (load balancer keeps instance in rotation)
        return JSONResponse(content=health_status, status_code=200)
    except Exception as e:
        # Unexpected error means service is unhealthy
        logger.error(f"Health check failed: {e}")
        # Return 503 Service Unavailable (load balancer removes from rotation)
        return JSONResponse(
            content={"status": "unhealthy", "error": str(e)}, 
            status_code=503
        )

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Chainlit Voice Chat Application", "status": "running"}

# Audio processing constants
SILENCE_THRESHOLD = 3500  # RMS energy threshold for silence detection
SILENCE_TIMEOUT = 1300.0  # Milliseconds of silence to end turn


@cl.step(type="tool")
async def speech_to_text(audio_buffer):
    """Convert audio buffer to text using AWS Transcribe
    
    Process:
    1. Upload audio to S3 (Transcribe requires S3 URI)
    2. Start async transcription job
    3. Poll for completion
    4. Download and parse results
    5. Clean up temporary files
    """
    # Generate unique identifiers to avoid naming conflicts
    job_name = f"transcribe-job-{uuid.uuid4()}"  # Unique job name
    s3_key = f"audio/{job_name}.wav"  # S3 path with audio/ prefix for organization
    
    try:
        # Step 1: Upload audio to S3
        # AWS Transcribe requires audio files to be in S3, can't process from memory
        logger.info(f"Uploading audio to S3: s3://{S3_BUCKET}/{s3_key}")
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,  # File path within bucket
            Body=audio_buffer,  # Raw audio bytes from WAV file
            ContentType='audio/wav'  # MIME type for proper handling
        )
        logger.info("Audio uploaded successfully to S3")
        
        # Verify the upload by checking if the object exists
        # This helps debug permission issues early
        try:
            s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)
            logger.info("Verified: uploaded audio file is accessible in S3")
        except Exception as verify_error:
            logger.error(f"Failed to verify uploaded file: {verify_error}")
            raise Exception(f"S3 upload verification failed: {verify_error}")
        
        # Step 2: Start transcription job
        # Transcribe jobs are asynchronous - we start them and poll for results
        logger.info(f"Starting transcription job: {job_name}")
        s3_uri = f's3://{S3_BUCKET}/{s3_key}'
        logger.info(f"Using S3 URI for Transcribe: {s3_uri}")
        
        try:
            transcribe_client.start_transcription_job(
                TranscriptionJobName=job_name,  # Must be unique across account
                Media={'MediaFileUri': s3_uri},  # S3 URI format required
                MediaFormat='wav',  # Tell Transcribe what audio format to expect
                LanguageCode='en-US'  # Language model to use for transcription
            )
            logger.info("Transcription job started successfully")
        except Exception as transcribe_error:
            logger.error(f"Failed to start transcription job: {transcribe_error}")
            # Add specific error details for common permission issues
            if "AccessDenied" in str(transcribe_error) or "BadRequest" in str(transcribe_error):
                logger.error("This likely indicates a permission issue with S3 access")
                logger.error("Ensure the ECS task role has s3:GetObject permission on the audio bucket")
            raise
        
        # Step 3: Poll for job completion
        # Transcription takes time (usually 10-30 seconds for short audio)
        while True:
            status = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
            job_status = status['TranscriptionJob']['TranscriptionJobStatus']
            
            # Job can be: IN_PROGRESS, COMPLETED, FAILED, or QUEUED
            if job_status in ['COMPLETED', 'FAILED']:
                break  # Exit polling loop
            
            # Wait 1 second before checking again (avoid rate limiting)
            await cl.sleep(1)
        
        # Step 4: Handle job failure
        if status['TranscriptionJob']['TranscriptionJobStatus'] == 'FAILED':
            # Extract specific failure reason from AWS response
            failure_reason = status['TranscriptionJob'].get('FailureReason', 'Unknown error')
            raise Exception(f"Transcription failed: {failure_reason}")
        
        # Step 5: Download and parse transcription results
        # AWS provides results as JSON file in S3
        transcript_uri = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
        import urllib.request  # Standard library for HTTP requests
        
        with urllib.request.urlopen(transcript_uri) as response:
            # Download JSON file containing transcription results
            transcript_data = json.loads(response.read())
        
        # Extract the actual transcribed text from nested JSON structure
        # Structure: results -> transcripts -> [0] -> transcript
        transcription = transcript_data['results']['transcripts'][0]['transcript']
        
        # Validate transcription is not empty
        # AWS Transcribe can return empty strings for silence or unclear audio
        if not transcription or not transcription.strip():
            logger.warning("Transcription resulted in empty text")
            raise Exception("No speech detected in audio. Please try speaking more clearly or loudly.")
        
        logger.info(f"Transcription completed successfully: '{transcription[:50]}...'")
        
        # Step 6: Clean up temporary S3 object to avoid storage costs
        s3_client.delete_object(Bucket=S3_BUCKET, Key=s3_key)
        logger.info("Cleaned up S3 object")
        
        return transcription.strip()  # Return cleaned transcription
        
    except Exception as e:
        logger.error(f"Speech to text failed: {e}")
        
        # Always attempt cleanup, even on failure
        # This prevents accumulating temporary files in S3
        try:
            s3_client.delete_object(Bucket=S3_BUCKET, Key=s3_key)
            logger.info("Cleaned up S3 object after error")
        except Exception as cleanup_error:
            # Cleanup failure is logged but not fatal
            logger.error(f"Failed to cleanup S3 object: {cleanup_error}")
        
        # Re-raise original exception to caller
        raise


@cl.step(type="tool")
async def text_to_speech(text: str, voice_id: str = "Joanna"):
    """Convert text to speech audio using AWS Polly
    
    Process:
    1. Send text to Polly with voice configuration
    2. Receive audio stream in response
    3. Read binary audio data
    4. Return filename and audio bytes
    """
    try:
        # Call AWS Polly to synthesize speech
        # Polly is synchronous (unlike Transcribe) - returns audio immediately
        response = polly_client.synthesize_speech(
            Text=text,  # Input text to convert to speech
            OutputFormat='mp3',  # Audio format (mp3 is compressed, good for web)
            VoiceId=voice_id,  # Voice to use (Joanna is female US English)
            Engine='neural'  # Neural engine sounds more natural than standard
        )
        
        # Read the audio stream from Polly response
        # Response contains 'AudioStream' which is a file-like object
        audio_data = response['AudioStream'].read()
        
        # Return tuple: (filename, audio_bytes)
        # Filename is for UI display, audio_bytes for playback
        return f"response_{uuid.uuid4()}.mp3", audio_data
        
    except Exception as e:
        # Common failures: invalid text, voice not available, API limits
        logger.error(f"Text to speech failed: {e}")
        raise


async def query_knowledge_base(query: str) -> str:
    """Query AWS Bedrock Knowledge Base for relevant information
    
    Args:
        query: User's question or search query
        
    Returns:
        Knowledge base response or empty string if disabled/failed
    """
    if not KNOWLEDGE_BASE_ENABLED or not KNOWLEDGE_BASE_ID or not bedrock_agent_client:
        return ""
    
    try:
        response = bedrock_agent_client.retrieve_and_generate(
            input={'text': query},
            retrieveAndGenerateConfiguration={
                'type': 'KNOWLEDGE_BASE',
                'knowledgeBaseConfiguration': {
                    'knowledgeBaseId': KNOWLEDGE_BASE_ID,
                    'modelArn': KNOWLEDGE_BASE_MODEL_ARN
                }
            }
        )
        
        return response.get('output', {}).get('text', '')
    except Exception as e:
        logger.warning(f"Knowledge base query failed: {e}")
        return ""


@cl.step(type="tool")
async def generate_text_answer(transcription):
    """Generate AI response using Claude via AWS Bedrock
    
    Process:
    1. Validate input transcription
    2. Get conversation history from session
    3. Add user's message to history
    4. Format messages for Claude API
    5. Call Bedrock with Claude model
    6. Parse response and update history
    7. Return AI's response text
    """
    # Step 1: Validate input
    if not transcription or not transcription.strip():
        logger.error("Empty transcription provided to generate_text_answer")
        raise ValueError("Cannot generate response for empty transcription")
    
    transcription = transcription.strip()  # Clean whitespace
    
    # Step 2: Retrieve and clean conversation history
    message_history = cl.user_session.get("message_history", [])
    
    # Step 3: Query knowledge base if enabled
    kb_context = await query_knowledge_base(transcription)
    
    # Step 4: Add user's transcribed message to conversation
    message_history.append({"role": "user", "content": transcription})
    
    # Step 5: Clean conversation history to ensure proper role alternation
    # This prevents consecutive messages with the same role
    cleaned_history = clean_conversation_history(message_history)
    
    # Step 6: Convert cleaned history to Claude API format
    claude_messages = []
    for msg in cleaned_history:
        if msg["role"] == "user":
            claude_messages.append({"role": "user", "content": msg["content"]})
        elif msg["role"] == "assistant":
            claude_messages.append({"role": "assistant", "content": msg["content"]})
    
    # Ensure we have at least one message for Claude
    if not claude_messages:
        logger.error("No valid messages to send to Claude after cleaning")
        # Fallback: create a single user message
        claude_messages = [{"role": "user", "content": transcription}]
    
    # Ensure the conversation starts with a user message (Claude requirement)
    if claude_messages[0]["role"] != "user":
        logger.info("Ensuring conversation starts with user message")
        claude_messages.insert(0, {"role": "user", "content": transcription})
    
    # Step 7: Prepare request body with optional knowledge base context
    system_prompt = "You are a helpful AI assistant. Provide clear, concise, and helpful responses."
    if kb_context:
        system_prompt += f"\n\nRelevant context from knowledge base:\n{kb_context}"
    
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "temperature": 0.2,
        "system": system_prompt,
        "messages": claude_messages
    }
    
    try:
        # Step 8: Call Claude via AWS Bedrock
        # Bedrock provides managed access to foundation models like Claude
        logger.info(f"Sending {len(claude_messages)} messages to Claude")
        
        # Debug: Log role sequence to verify alternation
        roles = [msg["role"] for msg in claude_messages]
        logger.info(f"Message role sequence: {' -> '.join(roles)}")
        
        # Verify role alternation before sending (final safety check)
        for i in range(1, len(claude_messages)):
            if claude_messages[i]["role"] == claude_messages[i-1]["role"]:
                logger.error(f"Role alternation violation detected at position {i}")
                raise ValueError(f"Invalid role sequence: {roles}")
        
        logger.debug(f"Claude messages: {claude_messages}")
        
        response = bedrock_client.invoke_model(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",  # Specific Claude model version
            body=json.dumps(body)  # Request must be JSON string
        )
        
        # Step 9: Parse Bedrock response
        # Bedrock wraps the model response in its own format
        response_body = json.loads(response['body'].read())
        
        # Extract actual text from Claude's response structure
        # Structure: content -> [0] -> text
        assistant_message = response_body['content'][0]['text']
        
        # Validate response is not empty
        if not assistant_message or not assistant_message.strip():
            logger.error("Claude returned empty response")
            assistant_message = "I apologize, but I couldn't generate a proper response. Please try again."
        
        # Step 10: Add AI response to conversation history
        # This maintains context for future turns in the conversation
        message_history.append({"role": "assistant", "content": assistant_message.strip()})
        
        # Clean and update session with properly alternating message history
        cleaned_message_history = clean_conversation_history(message_history)
        cl.user_session.set("message_history", cleaned_message_history)
        
        return assistant_message.strip()
        
    except Exception as e:
        # Common failures: model limits, malformed request, API errors, role alternation issues
        error_msg = str(e)
        logger.error(f"Generate text answer failed: {error_msg}")
        
        # Handle specific validation errors
        if "roles must alternate" in error_msg:
            logger.error("Role alternation error - clearing conversation history")
            # Reset conversation history and try with just the current message
            cl.user_session.set("message_history", [])
            simplified_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "temperature": 0.2,
                "system": "You are a helpful AI assistant. Provide clear, concise, and helpful responses.",
                "messages": [{"role": "user", "content": transcription}]
            }
            try:
                logger.info("Retrying with simplified message history")
                response = bedrock_client.invoke_model(
                    modelId="anthropic.claude-3-sonnet-20240229-v1:0",
                    body=json.dumps(simplified_body)
                )
                response_body = json.loads(response['body'].read())
                assistant_message = response_body['content'][0]['text']
                
                # Start fresh conversation history
                new_history = [
                    {"role": "user", "content": transcription},
                    {"role": "assistant", "content": assistant_message.strip()}
                ]
                cl.user_session.set("message_history", new_history)
                
                return assistant_message.strip()
            except Exception as retry_error:
                logger.error(f"Retry also failed: {retry_error}")
                return "I apologize, but I'm having trouble processing your request. Please try again."
        
        raise


@cl.on_chat_start
async def start():
    """Initialize new chat session
    
    Called when user opens the chat interface.
    Sets up session state and sends welcome message.
    """
    # Initialize empty conversation history for this session
    # Each user gets their own isolated conversation context
    cl.user_session.set("message_history", [])
    
    # Send welcome message with usage instructions
    # User needs to know how to activate voice input (press 'p')
    await cl.Message(
        content="Welcome to Chainlit x AWS example! Press `p` to talk!",
    ).send()


@cl.on_audio_start
async def on_audio_start():
    """Initialize audio recording session
    
    Called when user starts voice input (presses 'p').
    Resets all audio processing state variables.
    """
    # Reset silence detection state
    cl.user_session.set("silent_duration_ms", 0)  # How long user has been silent
    cl.user_session.set("is_speaking", False)  # Whether user is currently speaking
    
    # Initialize audio buffer for collecting chunks
    cl.user_session.set("audio_chunks", [])  # List to store numpy arrays of audio data
    
    # Return True to allow audio recording to proceed
    return True


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk):
    """Process individual audio chunks during recording
    
    Called continuously while user is recording audio.
    Implements voice activity detection to automatically end recording
    when user stops speaking.
    
    Algorithm:
    1. Store audio chunk for later processing
    2. Calculate audio energy (volume level)
    3. Track silence duration
    4. Trigger processing when silence threshold exceeded
    """
    # Step 1: Store audio chunk
    audio_chunks = cl.user_session.get("audio_chunks")
    
    if audio_chunks is not None:
        # Convert raw audio bytes to numpy array for processing
        # dtype=np.int16 because audio is 16-bit PCM format
        audio_chunk = np.frombuffer(chunk.data, dtype=np.int16)
        audio_chunks.append(audio_chunk)  # Add to collection for later concatenation

    # Handle first chunk: initialize timing variables
    if chunk.isStart:
        # Set baseline timestamp for calculating time differences
        cl.user_session.set("last_elapsed_time", chunk.elapsedTime)
        cl.user_session.set("is_speaking", True)  # Assume user starts speaking
        return  # Skip voice activity detection on first chunk

    # Step 2: Get current session state
    audio_chunks = cl.user_session.get("audio_chunks")
    last_elapsed_time = cl.user_session.get("last_elapsed_time")
    silent_duration_ms = cl.user_session.get("silent_duration_ms")
    is_speaking = cl.user_session.get("is_speaking")

    # Step 3: Calculate time elapsed since last chunk
    # This helps track how long user has been silent
    time_diff_ms = chunk.elapsedTime - last_elapsed_time
    cl.user_session.set("last_elapsed_time", chunk.elapsedTime)

    # Step 4: Calculate audio energy using RMS (Root Mean Square)
    # RMS gives us the "loudness" or energy level of the audio
    audio_chunk = np.frombuffer(chunk.data, dtype=np.int16)
    # Convert to float32 to avoid integer overflow in calculations
    audio_energy = np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2))

    # Step 5: Voice Activity Detection
    if audio_energy < SILENCE_THRESHOLD:
        # Audio is considered silent (below threshold)
        # Accumulate silence duration
        silent_duration_ms += time_diff_ms
        cl.user_session.set("silent_duration_ms", silent_duration_ms)
        
        # Check if we've been silent long enough to end the turn
        if silent_duration_ms >= SILENCE_TIMEOUT and is_speaking:
            cl.user_session.set("is_speaking", False)  # Mark as stopped speaking
            await process_audio()  # Process accumulated audio
    else:
        # Audio is not silent (above threshold)
        # Reset silence counter and mark as speaking
        cl.user_session.set("silent_duration_ms", 0)
        if not is_speaking:
            cl.user_session.set("is_speaking", True)  # Mark as started speaking


async def process_audio():
    """Process accumulated audio chunks and handle complete voice interaction
    
    Called when voice activity detection determines user has stopped speaking.
    
    Process:
    1. Combine audio chunks into single WAV file
    2. Validate audio duration
    3. Convert speech to text
    4. Generate AI response
    5. Convert response to speech
    6. Display both messages in UI
    """
    # Step 1: Get accumulated audio chunks
    if audio_chunks := cl.user_session.get("audio_chunks"):
        
        # Step 2: Combine all audio chunks into continuous stream
        # Each chunk is a numpy array, concatenate creates single array
        concatenated = np.concatenate(list(audio_chunks))

        # Step 3: Create WAV file in memory
        # AWS Transcribe requires proper WAV format with headers
        wav_buffer = io.BytesIO()  # In-memory binary buffer
        
        with wave.open(wav_buffer, "wb") as wav_file:
            # Configure WAV file parameters to match input audio
            wav_file.setnchannels(1)      # Mono audio (single channel)
            wav_file.setsampwidth(2)      # 16-bit audio (2 bytes per sample)
            wav_file.setframerate(24000)  # 24kHz sample rate
            
            # Write audio data as bytes to WAV file
            wav_file.writeframes(concatenated.tobytes())

        # Reset buffer position to beginning for reading
        wav_buffer.seek(0)
        
        # Clear audio chunks to prepare for next recording
        cl.user_session.set("audio_chunks", [])

        # Step 4: Validate audio duration
        # Very short audio often contains no useful speech
        frames = wav_file.getnframes()    # Total audio frames
        rate = wav_file.getframerate()    # Sample rate (frames per second)
        duration = frames / float(rate)   # Duration in seconds
        
        # if duration <= 1.71:  # Minimum duration threshold
        #     print("The audio is too short, please try again.")
        #     return  # Exit without processing

        # Step 5: Get final audio data for processing
        audio_buffer = wav_buffer.getvalue()  # Extract bytes from buffer
        
        # Create UI element to display user's audio input
        input_audio_el = cl.Audio(content=audio_buffer, mime="audio/wav")

        # Step 6: Process complete voice interaction pipeline
        try:
            # Convert speech to text using AWS Transcribe
            transcription = await speech_to_text(audio_buffer)
            
            # Display user's message with transcription and playable audio
            await cl.Message(
                author="You",                    # Show as user's message
                type="user_message",            # Message type for styling
                content=transcription,          # Display transcribed text
                elements=[input_audio_el],      # Include audio playback
            ).send()

            # Generate AI response from transcribed text
            answer = await generate_text_answer(transcription)
            
            # Convert AI response to speech audio
            output_name, output_audio = await text_to_speech(answer)

            # Create audio element for AI response
            output_audio_el = cl.Audio(
                auto_play=True,        # Automatically play when message appears
                mime="audio/mp3",      # MP3 format from Polly
                content=output_audio,  # Audio bytes
            )

            # Display AI response with text and auto-playing audio
            await cl.Message(content=answer, elements=[output_audio_el]).send()
            
        except Exception as e:
            logger.error(f"Voice processing pipeline failed: {e}")
            
            # Show user-friendly error message
            error_message = "Sorry, I couldn't process your voice input. "
            
            if "No speech detected" in str(e):
                error_message += "Please try speaking more clearly or loudly."
            elif "Empty transcription" in str(e) or "ValidationException" in str(e):
                error_message += "I didn't detect any speech. Please try again."
            elif "BadRequest" in str(e) or "AccessDenied" in str(e):
                error_message += "There was a technical issue. Please try again later."
            else:
                error_message += "Please try again."
            
            # Display error with the original audio for debugging
            await cl.Message(
                content=error_message,
                elements=[input_audio_el],  # Include audio so user can verify what was recorded
            ).send()


def clean_conversation_history(message_history):
    """Clean conversation history to ensure proper role alternation
    
    Claude requires strict alternation between user and assistant roles.
    This function removes consecutive messages with the same role.
    
    Args:
        message_history: List of message dictionaries with 'role' and 'content'
    
    Returns:
        List of cleaned messages with proper alternation
    """
    if not message_history:
        return []
    
    cleaned_history = []
    last_role = None
    
    for msg in message_history:
        content = msg["content"].strip() if msg["content"] else ""
        if not content:
            continue  # Skip empty messages
            
        current_role = msg["role"]
        
        # Only add if role is different from previous or if it's the first message
        if current_role != last_role:
            cleaned_history.append({"role": current_role, "content": content})
            last_role = current_role
        else:
            # Replace the last message of the same role (keep the most recent)
            cleaned_history[-1] = {"role": current_role, "content": content}
    
    return cleaned_history


@cl.on_message
async def on_message(message: cl.Message):
    """Handle text messages (redirect to voice interface)
    
    This app is designed for voice interaction only.
    Any text messages are redirected with instructions to use voice.
    """
    # Inform user this is voice-only and provide usage instructions
    await cl.Message(content="This is a voice demo, press P to start!").send()
