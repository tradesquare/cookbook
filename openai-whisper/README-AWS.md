# Chainlit x AWS Voice Chat

This application uses AWS services instead of OpenAI:
- **AWS Transcribe** for speech-to-text (replaces OpenAI Whisper)
- **AWS Bedrock Claude** for text generation (replaces OpenAI GPT)
- **ElevenLabs** for text-to-speech (unchanged)

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements-aws.txt
```

### 2. AWS Configuration
Configure your AWS credentials using one of these methods:
- AWS CLI: `aws configure`
- Environment variables: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
- IAM roles (if running on EC2)

### 3. Required AWS Services
Enable these services in your AWS account:
- **Amazon Transcribe**
- **Amazon Bedrock** (with Claude model access)
- **Amazon S3** (for temporary audio storage)

### 4. Environment Variables
```bash
export ELEVENLABS_API_KEY="your_elevenlabs_api_key"
export ELEVENLABS_VOICE_ID="your_voice_id"
export AWS_REGION="us-east-1"  # Optional, defaults to us-east-1
export S3_BUCKET="your-transcribe-bucket"  # S3 bucket for temp audio files
```

### 5. S3 Bucket Setup
Create an S3 bucket for temporary audio file storage:
```bash
aws s3 mb s3://your-transcribe-bucket
```

### 6. IAM Permissions
Your AWS user/role needs these permissions:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "transcribe:StartTranscriptionJob",
                "transcribe:GetTranscriptionJob"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel"
            ],
            "Resource": "arn:aws:bedrock:*::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0"
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:PutObject",
                "s3:DeleteObject"
            ],
            "Resource": "arn:aws:s3:::your-transcribe-bucket/*"
        }
    ]
}
```

## Run the Application
```bash
chainlit run app.py
```

## Key Changes from OpenAI Version

1. **Speech-to-Text**: Uses AWS Transcribe instead of OpenAI Whisper
   - Uploads audio to S3 temporarily
   - Creates transcription job
   - Polls for completion
   - Cleans up S3 object

2. **Text Generation**: Uses AWS Bedrock Claude instead of OpenAI GPT
   - Converts message format for Claude
   - Uses Claude 3 Sonnet model
   - Maintains conversation history

3. **Dependencies**: Uses `boto3` instead of `openai` package

## Cost Considerations

- **AWS Transcribe**: ~$0.024 per minute of audio
- **AWS Bedrock Claude**: ~$0.003 per 1K input tokens, ~$0.015 per 1K output tokens
- **S3**: Minimal cost for temporary storage
- **ElevenLabs**: As per your ElevenLabs plan