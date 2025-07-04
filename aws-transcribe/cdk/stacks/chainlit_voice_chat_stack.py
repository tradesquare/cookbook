# Import AWS CDK core modules and service-specific constructs
from aws_cdk import (
    Stack,  # Base class for CDK stacks
    aws_ec2 as ec2,  # VPC, subnets, security groups
    aws_ecs as ecs,  # Elastic Container Service for running containers
    aws_iam as iam,  # Identity and Access Management for permissions
    aws_s3 as s3,  # Simple Storage Service for file storage
    aws_logs as logs,  # CloudWatch Logs for application logging
    aws_elasticloadbalancingv2 as elbv2,  # Application Load Balancer
    aws_certificatemanager as acm,  # For SSL/TLS certificates
    aws_secretsmanager as secretsmanager,  # For storing sensitive data (imported but not used)
    RemovalPolicy,  # Defines what happens to resources when stack is deleted
    Duration,  # Helper for time-based configurations
    CfnOutput  # For outputting stack values (imported but not used)
)
from constructs import Construct  # Base class for all CDK constructs


class ChainlitVoiceChatStack(Stack):
    """CDK Stack for deploying a Chainlit voice chat application on AWS ECS Fargate
    
    This stack creates:
    - VPC with public subnets for internet access
    - S3 bucket for temporary audio file storage
    - ECS Fargate cluster and service for running the Chainlit app
    - Application Load Balancer for distributing traffic
    - IAM roles with permissions for AWS services (Transcribe, Bedrock, Polly)
    """
    
    def __init__(self, scope: Construct, construct_id: str, env_name: str, certificate_arn: str = None, **kwargs) -> None:
        """Initialize the stack with required AWS resources
        
        Args:
            scope: The parent construct (usually the CDK app)
            construct_id: Unique identifier for this stack (e.g., 'ChainlitVoiceChatStack')
            env_name: Environment name for resource naming (e.g., 'dev', 'prod')
            certificate_arn: ARN of existing ACM certificate for HTTPS
            **kwargs: Additional CDK stack arguments (region, account, etc.)
        """
        super().__init__(scope, construct_id, **kwargs)
        
        # Store environment name for resource naming consistency
        # Example: if env_name='dev', resources will be named like 'chainlit-voice-chat-dev'
        self.env_name = env_name
        self.certificate_arn = certificate_arn
        
        # Create VPC first as other resources depend on it
        # VPC provides isolated network environment for all resources
        self.vpc = self._create_vpc()
        
        # Create S3 bucket for temporary audio file storage
        # Used by the app to store uploaded audio files before processing
        self.audio_bucket = self._create_s3_bucket()
        
        # Create IAM role with permissions for AWS AI services
        # ECS tasks will assume this role to access Transcribe, Bedrock, Polly
        self.task_role = self._create_task_role()
        
        # Create ECS cluster (compute environment) and service (running containers)
        # Cluster manages the infrastructure, service ensures containers stay running
        self.cluster = self._create_ecs_cluster()
        self.service = self._create_ecs_service()
        
        # Create Application Load Balancer for public internet access
        # Routes HTTP traffic from internet to ECS containers
        self.alb = self._create_load_balancer()
        
        # Create CloudFormation outputs for important resource information
        # (Method called but not implemented in current code)
        self._create_outputs()
    
    def _create_vpc(self) -> ec2.Vpc:
        """Create VPC with public subnets for cost-effective deployment
        
        Returns:
            ec2.Vpc: VPC with public subnets across 2 availability zones
        """
        return ec2.Vpc(
            self, "VPC",  # Construct ID within this stack
            max_azs=2,  # Use 2 availability zones for high availability (e.g., us-east-1a, us-east-1b)
            nat_gateways=0,  # No NAT gateways to reduce costs (containers will be in public subnets)
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="Public",  # Subnet name prefix
                    subnet_type=ec2.SubnetType.PUBLIC,  # Public subnet with internet gateway access
                    cidr_mask=24  # /24 CIDR block = 256 IP addresses per subnet (e.g., 10.0.1.0/24)
                ),
                # Private subnets commented out to reduce costs
                # Would require NAT gateway ($45/month) for internet access
                # ec2.SubnetConfiguration(
                #     name="Private",
                #     subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,  # Private with NAT gateway
                #     cidr_mask=24  # /24 CIDR block for private subnets
                # )
            ]
        )
    
    def _create_s3_bucket(self) -> s3.Bucket:
        """Create S3 bucket for temporary audio file storage with automatic cleanup
        
        Returns:
            s3.Bucket: S3 bucket configured for temporary audio storage
        """
        bucket = s3.Bucket(
            self, "AudioBucket",  # Construct ID
            # Unique bucket name: chainlit-voice-chat-audio-dev-123456789012
            bucket_name=f"chainlit-voice-chat-audio-{self.env_name}-{self.account}",
            removal_policy=RemovalPolicy.DESTROY,  # Delete bucket when stack is destroyed
            auto_delete_objects=True,  # Automatically delete all objects before bucket deletion
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="DeleteTempAudio",  # Rule identifier
                    expiration=Duration.days(1),  # Delete objects after 1 day to save storage costs
                    enabled=True  # Enable this lifecycle rule
                )
            ],
            cors=[
                # Cross-Origin Resource Sharing rules for web browser access
                s3.CorsRule(
                    # Allow GET (download), PUT (upload), DELETE operations
                    allowed_methods=[s3.HttpMethods.GET, s3.HttpMethods.PUT, s3.HttpMethods.DELETE],
                    allowed_origins=["*"],  # Allow requests from any domain (use specific domains in production)
                    allowed_headers=["*"]  # Allow any headers in requests
                )
            ]
        )
        return bucket
    
    def _create_task_role(self) -> iam.Role:
        """Create IAM role for ECS task with permissions for AWS AI services
        
        The role allows the containerized application to:
        - Store/delete audio files in S3
        - Use Amazon Transcribe for speech-to-text
        - Use Amazon Bedrock (Claude) for AI responses
        - Use Amazon Polly for text-to-speech
        
        Returns:
            iam.Role: IAM role with necessary permissions
        """
        role = iam.Role(
            self, "TaskRole",  # Construct ID
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),  # ECS tasks can assume this role
            description="Role for Chainlit Voice Chat ECS task"  # Human-readable description
        )
        
        # S3 permissions for audio file operations
        # Allows uploading user audio, reading files for Transcribe, and deleting processed files
        role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,  # Grant permission (vs DENY)
                actions=[
                    "s3:PutObject",    # Upload audio files (e.g., user_audio_123.wav)
                    "s3:GetObject",    # Read audio files (required for Transcribe to access S3 URIs)
                    "s3:DeleteObject"  # Delete processed files to save storage
                ],
                # Only allow access to objects in our specific bucket
                # Example: arn:aws:s3:::chainlit-voice-chat-audio-dev-123456789012/*
                resources=[f"{self.audio_bucket.bucket_arn}/*"]
            )
        )
        
        # Additional S3 bucket-level permissions
        # Required for certain S3 operations and service integrations
        role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "s3:ListBucket",        # List bucket contents (useful for debugging)
                    "s3:GetBucketLocation"  # Get bucket region (some AWS services require this)
                ],
                # Bucket-level permissions (no /* suffix)
                resources=[self.audio_bucket.bucket_arn]
            )
        )
        
        # Amazon Transcribe permissions for speech-to-text conversion
        # Supports both batch transcription and real-time streaming
        role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "transcribe:StartTranscriptionJob",  # Start async transcription job
                    "transcribe:GetTranscriptionJob",    # Check job status and get results
                    "transcribe:DeleteTranscriptionJob", # Clean up completed jobs (optional)
                    "transcribe:StartStreamTranscription", # Start real-time streaming transcription
                    "transcribe:StartStreamTranscriptionWebSocket" # Start WebSocket streaming transcription
                ],
                resources=["*"]  # Transcribe jobs don't have specific ARNs
            )
        )
        
        # Amazon Bedrock permissions for AI model inference
        # Uses Claude 3.5 Sonnet for generating intelligent responses
        role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "bedrock:InvokeModel"  # Call the AI model with prompts
                ],
                resources=[
                    # Claude 3.5 Sonnet model ARN (used in application)
                    f"arn:aws:bedrock:{self.region}::foundation-model/anthropic.claude-3-5-sonnet-20240620-v1:0",
                    # Claude 3 Sonnet model ARN (fallback)
                    f"arn:aws:bedrock:{self.region}::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0"
                ]
            )
        )
        
        # Amazon Polly permissions for text-to-speech conversion
        # Converts AI responses back to audio for voice chat
        role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "polly:SynthesizeSpeech"  # Convert text to audio (e.g., MP3, WAV)
                ],
                resources=["*"]  # Polly doesn't use resource-specific ARNs
            )
        )
        
        return role
    
    def _create_ecs_cluster(self) -> ecs.Cluster:
        """Create ECS cluster for running containerized applications
        
        ECS Cluster is a logical grouping of compute resources (EC2 or Fargate)
        where containers run. This uses Fargate (serverless) so no EC2 management needed.
        
        Returns:
            ecs.Cluster: ECS cluster for running containers
        """
        return ecs.Cluster(
            self, "Cluster",  # Construct ID
            vpc=self.vpc,  # Deploy cluster in our VPC network
            # Cluster name example: chainlit-voice-chat-dev
            cluster_name=f"chainlit-voice-chat-{self.env_name}"
        )
    
    def _create_ecs_service(self) -> ecs.FargateService:
        """Create ECS Fargate service with container definition and networking
        
        Creates:
        - CloudWatch log group for application logs
        - Task definition (container blueprint)
        - Container with environment variables
        - Security group for network access
        - Fargate service to run containers
        
        Returns:
            ecs.FargateService: Running ECS service
        """
        # Create CloudWatch log group for application logs
        # All container stdout/stderr will be sent here
        log_group = logs.LogGroup(
            self, "LogGroup",  # Construct ID
            # Log group name example: /ecs/chainlit-voice-chat-dev
            log_group_name=f"/ecs/chainlit-voice-chat-{self.env_name}",
            removal_policy=RemovalPolicy.DESTROY,  # Delete logs when stack is destroyed
            retention=logs.RetentionDays.ONE_WEEK  # Keep logs for 7 days to control costs
        )
        
        # Create task definition (blueprint for containers)
        # Defines CPU, memory, and container specifications
        task_definition = ecs.FargateTaskDefinition(
            self, "TaskDefinition",  # Construct ID
            memory_limit_mib=2048,  # 2GB RAM (sufficient for AI processing)
            cpu=1024,  # 1 vCPU (1024 CPU units)
            task_role=self.task_role  # IAM role for AWS service access
        )
        
        # Add container to task definition
        # This is the actual application container that will run
        container = task_definition.add_container(
            "ChainlitContainer",  # Container name
            # Build Docker image from parent directory (../)
            # Expects Dockerfile in the parent directory
            image=ecs.ContainerImage.from_asset(".."),
            # Configure CloudWatch logging
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix="chainlit",  # Log stream prefix
                log_group=log_group  # Send logs to our log group
            ),
            # Environment variables passed to container
            environment={
                "AWS_REGION": self.region,  # Current AWS region (e.g., us-east-1)
                "S3_BUCKET": self.audio_bucket.bucket_name,  # S3 bucket for audio files
                "CHAINLIT_HOST": "0.0.0.0",  # Listen on all interfaces
                "CHAINLIT_PORT": "8000"  # Application port
            },
            # Port mapping for network access
            port_mappings=[
                ecs.PortMapping(
                    container_port=8000,  # Port inside container
                    protocol=ecs.Protocol.TCP  # TCP protocol for HTTP
                )
            ]
        )
        
        # Create security group for ECS service (network firewall rules)
        # Controls what traffic can reach the containers
        security_group = ec2.SecurityGroup(
            self, "EcsSecurityGroup",  # Construct ID
            vpc=self.vpc,  # Deploy in our VPC
            description="Security group for Chainlit ECS service",
            allow_all_outbound=True  # Allow containers to make outbound requests (for AWS APIs)
        )
        
        # Allow inbound HTTP traffic on port 8000
        # This allows the load balancer to reach the containers
        security_group.add_ingress_rule(
            peer=ec2.Peer.any_ipv4(),  # Allow from any IP (0.0.0.0/0)
            connection=ec2.Port.tcp(8000),  # TCP port 8000
            description="Allow HTTP traffic"  # Rule description
        )
        
        # Create Fargate service to run and manage containers
        # Ensures desired number of containers are always running
        service = ecs.FargateService(
            self, "Service",  # Construct ID
            cluster=self.cluster,  # Run in our ECS cluster
            task_definition=task_definition,  # Use our container blueprint
            desired_count=1,  # Run 1 container instance
            security_groups=[security_group],  # Apply our security group
            # Deploy in public subnets since private subnets are not available
            vpc_subnets=ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.PUBLIC  # Public subnets
            ),
            assign_public_ip=True  # Assign public IP for public subnet deployment
        )
        
        return service
    
    def _create_load_balancer(self) -> elbv2.ApplicationLoadBalancer:
        """Create Application Load Balancer for public internet access
        
        Creates:
        - Security group for ALB with HTTP/HTTPS access
        - Internet-facing Application Load Balancer
        - Target group for ECS containers
        - Health checks to ensure containers are healthy
        - HTTP listener to route traffic
        
        Returns:
            elbv2.ApplicationLoadBalancer: Load balancer for public access
        """
        # Create security group for Application Load Balancer
        # Controls what traffic can reach the load balancer from internet
        alb_security_group = ec2.SecurityGroup(
            self, "AlbSecurityGroup",  # Construct ID
            vpc=self.vpc,  # Deploy in our VPC
            description="Security group for Application Load Balancer",
            allow_all_outbound=True  # Allow ALB to reach ECS containers
        )
        
        # Allow HTTP traffic from internet (port 80)
        # Users can access the app via http://your-alb-dns-name.com
        alb_security_group.add_ingress_rule(
            peer=ec2.Peer.any_ipv4(),  # Allow from anywhere on internet (0.0.0.0/0)
            connection=ec2.Port.tcp(80),  # Standard HTTP port
            description="Allow HTTP traffic"  # Rule description
        )
        
        # Allow HTTPS traffic from internet (port 443)
        # For future SSL/TLS certificate implementation
        alb_security_group.add_ingress_rule(
            peer=ec2.Peer.any_ipv4(),  # Allow from anywhere on internet
            connection=ec2.Port.tcp(443),  # Standard HTTPS port
            description="Allow HTTPS traffic"  # Rule description
        )
        
        # Create Application Load Balancer
        # Distributes incoming traffic across healthy ECS containers
        alb = elbv2.ApplicationLoadBalancer(
            self, "LoadBalancer",  # Construct ID
            vpc=self.vpc,  # Deploy in our VPC
            internet_facing=True,  # Accessible from internet (vs internal)
            security_group=alb_security_group,  # Apply our security group
            # Deploy in public subnets so it can receive internet traffic
            vpc_subnets=ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.PUBLIC  # Public subnets with internet gateway
            )
        )
        
        # Create target group to define how to route traffic to containers
        # Target group contains the ECS containers that will receive traffic
        target_group = elbv2.ApplicationTargetGroup(
            self, "TargetGroup",  # Construct ID
            port=8000,  # Port where containers are listening
            protocol=elbv2.ApplicationProtocol.HTTP,  # HTTP protocol
            vpc=self.vpc,  # Deploy in our VPC
            target_type=elbv2.TargetType.IP,  # Target containers by IP (Fargate mode)
            # Health check configuration to ensure containers are working
            health_check=elbv2.HealthCheck(
                enabled=True,  # Enable health checks
                healthy_http_codes="200",  # Consider HTTP 200 as healthy
                path="/",  # Check main Chainlit endpoint
                timeout=Duration.seconds(10),  # Wait 10 seconds for response
                interval=Duration.seconds(30),  # Check every 30 seconds
                healthy_threshold_count=2,  # 2 consecutive successes = healthy
                unhealthy_threshold_count=5  # 5 consecutive failures = unhealthy
            )
        )
        
        # Register ECS service containers as targets
        # This connects the load balancer to the running containers
        self.service.attach_to_application_target_group(target_group)
        
        # Add HTTPS listener if certificate is provided (keep existing HTTP listener)
        if self.certificate_arn:
            certificate = acm.Certificate.from_certificate_arn(
                self, "Certificate",
                certificate_arn=self.certificate_arn
            )
            
            alb.add_listener(
                "HttpsListener",
                port=443,
                protocol=elbv2.ApplicationProtocol.HTTPS,
                certificates=[certificate],
                default_target_groups=[target_group]
            )
        
        # Only create HTTP listener if it doesn't exist (for new deployments)
        if not self.certificate_arn:
            alb.add_listener(
                "Listener",
                port=80,
                protocol=elbv2.ApplicationProtocol.HTTP,
                default_target_groups=[target_group]
            )
        
        # Allow ALB to communicate with ECS service
        # (This comment indicates missing security group rule - should be implemented)
        self.service.connections.allow_from(
            alb,
            ec2.Port.tcp(8000),
            "Allow ALB to reach ECS service"
        )
        
        return alb
    
    def _create_outputs(self):
        """Create CloudFormation outputs"""
        CfnOutput(
            self, "LoadBalancerDNS",
            value=self.alb.load_balancer_dns_name,
            description="DNS name of the load balancer"
        )
        
        CfnOutput(
            self, "S3BucketName",
            value=self.audio_bucket.bucket_name,
            description="Name of the S3 bucket for audio files"
        )
        
        CfnOutput(
            self, "ApplicationURL",
            value=f"{'https' if self.certificate_arn else 'http'}://{self.alb.load_balancer_dns_name}",
            description="URL to access the Chainlit application"
        )