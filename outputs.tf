output "instance_public_ip" {
  value = aws_instance.rl_instance.public_ip
}

output "alb_dns_name" {
  value = aws_lb.app_lb.dns_name
}

output "lambda_function_name" {
  value = aws_lambda_function.load_generator.function_name
}
