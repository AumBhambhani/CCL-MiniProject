variable "aws_region" {
  default = "us-east-1"
}

variable "vpc_cidr" {
  default = "10.0.0.0/16"
}

variable "subnet_cidr" {
  default = "10.0.1.0/24"
}

variable "availability_zone" {
  default = "us-east-1a"
}

variable "subnet_cidr2" {
  default = "10.0.2.0/24"
}

variable "availability_zone2" {
  default = "us-east-1b"
}

variable "ami_id" {
  description = "AMI ID for EC2 instances (Ubuntu recommended)"
  type        = string
}

variable "instance_type" {
  default = "t2.micro"
}
