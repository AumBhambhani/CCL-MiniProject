#terraform init just downloads all the plugins required to interact with the respective provider.
#terraform plan is used to create an execution plan.
#terraform apply is used to apply the changes required to reach the desired state of the configuration.
#terraform destroy is used to destroy the Terraform-managed infrastructure.
#most of the times we do not need to specify every resource in order. Terraform will figure out the dependencies and create the resources in the correct order.
#Terraform is a declarative language, which means that you specify the desired state of the infrastructure, and Terraform will figure out how to create it.
#Terraform is an intelligent tool that can figure out the dependencies between resources and create them in the correct order.

provider "aws" {
  region = var.aws_region
  access_key = ""
  secret_key = ""
}

# VPC
resource "aws_vpc" "main" {
  cidr_block = var.vpc_cidr
}

# Subnet1
resource "aws_subnet" "public_subnet" {
  vpc_id            = aws_vpc.main.id
  cidr_block        = var.subnet_cidr
  availability_zone = var.availability_zone
}

#Subnet2
resource "aws_subnet" "public_subnet2" {
  vpc_id            = aws_vpc.main.id
  cidr_block        = var.subnet_cidr2
  availability_zone = var.availability_zone2
}

# Security Group
resource "aws_security_group" "ec2_sg" {
  vpc_id = aws_vpc.main.id

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# EC2 Instance for RL agent or backend
resource "aws_instance" "rl_instance" {
  ami           = var.ami_id
  instance_type = var.instance_type
  subnet_id     = aws_subnet.public_subnet.id
  security_groups = [aws_security_group.ec2_sg.name]

  tags = {
    Name = "RL-LoadBalancer-Agent"
  }
}

# S3 Bucket for storing Lambda function
resource "random_id" "bucket_suffix" {
  byte_length = 4
}

resource "aws_s3_bucket" "lambda_bucket" {
  bucket = "rl-load-gen-bucket-${random_id.bucket_suffix.hex}"
}

# Lambda Function for Load Generation
resource "aws_lambda_function" "load_generator" {
  function_name = "LoadGenerator"
  s3_bucket     = aws_s3_bucket.lambda_bucket.bucket
  s3_key        = "load_generator.zip"
  runtime       = "python3.8"
  handler       = "load_generator.handler"
  role          = aws_iam_role.lambda_exec_role.arn
}

# IAM Role for Lambda execution
resource "aws_iam_role" "lambda_exec_role" {
  name = "lambda_exec_role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
      Effect = "Allow"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_basic_execution" {
  role       = aws_iam_role.lambda_exec_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# (Optional) Application Load Balancer
resource "aws_lb" "app_lb" {
  name               = "rl-app-lb"
  internal           = false
  load_balancer_type = "application"
  subnets            = [aws_subnet.public_subnet.id, aws_subnet.public_subnet2.id]
  security_groups    = [aws_security_group.ec2_sg.id]
}

# Target group for backend
resource "aws_lb_target_group" "target_group" {
  name     = "rl-target-group"
  port     = 80
  protocol = "HTTP"
  vpc_id   = aws_vpc.main.id
  health_check {
    path                = "/"
    protocol            = "HTTP"
    interval            = 30
    timeout = 5
    healthy_threshold   = 3
    unhealthy_threshold = 3
  }
}

# Attach EC2 instance to ALB target group
resource "aws_lb_target_group_attachment" "ec2_attach" {
  target_group_arn = aws_lb_target_group.target_group.arn
  target_id        = aws_instance.rl_instance.id
  port             = 80
}

# ALB Listener
resource "aws_lb_listener" "http_listener" {
  load_balancer_arn = aws_lb.app_lb.arn
  port              = "80"
  protocol          = "HTTP"
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.target_group.arn
  }
}

# Internet Gateway for public access
resource "aws_internet_gateway" "igw" {
  vpc_id = aws_vpc.main.id
}

# Route table for public subnets
resource "aws_route_table" "public_rt" {
  vpc_id = aws_vpc.main.id
  
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.igw.id
  }
}

# Associate route table with public subnets
resource "aws_route_table_association" "public_subnet_association" {
  subnet_id      = aws_subnet.public_subnet.id
  route_table_id = aws_route_table.public_rt.id
}

resource "aws_route_table_association" "public_subnet2_association" {
  subnet_id      = aws_subnet.public_subnet2.id
  route_table_id = aws_route_table.public_rt.id
}