#!/bin/bash

# Local testing script for Chainlit Voice Chat application

set -e

echo "🧪 Testing Chainlit Voice Chat application locally..."

# Check if required environment variables are set
if [ -z "$AWS_REGION" ]; then
    export AWS_REGION="us-east-1"
    echo "⚠️  AWS_REGION not set, using default: $AWS_REGION"
fi

if [ -z "$S3_BUCKET" ]; then
    echo "❌ S3_BUCKET environment variable is required for local testing"
    echo "   Please set it to an existing S3 bucket name:"
    echo "   export S3_BUCKET=your-test-bucket-name"
    exit 1
fi

# Check if AWS credentials are configured
if ! aws sts get-caller-identity > /dev/null 2>&1; then
    echo "❌ AWS CLI is not configured. Please run 'aws configure' first."
    exit 1
fi

# Check if S3 bucket exists
if ! aws s3 ls "s3://$S3_BUCKET" > /dev/null 2>&1; then
    echo "❌ S3 bucket '$S3_BUCKET' does not exist or is not accessible"
    echo "   Creating bucket..."
    aws s3 mb "s3://$S3_BUCKET" --region "$AWS_REGION"
    echo "✅ Bucket created successfully"
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements-aws.txt

# Check if Bedrock model access is enabled
echo "🔍 Checking Bedrock model access..."
if aws bedrock list-foundation-models --region "$AWS_REGION" > /dev/null 2>&1; then
    echo "✅ Bedrock access confirmed"
else
    echo "⚠️  Bedrock access may not be enabled. Please ensure you have access to Bedrock models."
fi

# Run the application locally
echo "🚀 Starting Chainlit application..."
echo "   Application will be available at: http://localhost:8000"
echo "   Press Ctrl+C to stop"
echo ""

chainlit run app.py --host localhost --port 8000