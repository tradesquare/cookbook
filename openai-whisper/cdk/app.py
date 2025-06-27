#!/usr/bin/env python3
import aws_cdk as cdk
from stacks.chainlit_voice_chat_stack import ChainlitVoiceChatStack

app = cdk.App()

# Get environment variables or use defaults
# account = app.node.try_get_context("account") or "133021490917"
# region = app.node.try_get_context("region") or "us-east-1"
# env_name = app.node.try_get_context("environment") or "dev"
account = app.node.try_get_context("account") or "654654383273"
region = app.node.try_get_context("region") or "us-east-1"
env_name = app.node.try_get_context("environment") or "dev"
certificate_arn = app.node.try_get_context("certificate_arn")

env = cdk.Environment(account=account, region=region)

ChainlitVoiceChatStack(
    app, 
    f"ChainlitVoiceChat-{env_name}",
    env=env,
    env_name=env_name,
    certificate_arn=certificate_arn,
    description="Chainlit Voice Chat application with AWS services"
)

app.synth()