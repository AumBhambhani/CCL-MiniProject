import json
import asyncio
import aiohttp
import os
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Get ALB URL from environment variable
ALB_URL = os.environ.get('ALB_URL', 'http://my-load-balancer.amazonaws.com')

async def make_request(session, request_id):
    """Make a single request to the ALB"""
    try:
        # Add some randomness to the load
        delay_param = random.random() / 2
        
        start_time = time.time()
        async with session.get(
            ALB_URL,
            timeout=5,
            headers={"X-Load-Test": "true", "X-Request-ID": str(request_id)},
            params={"delay": delay_param}
        ) as response:
            response_time = time.time() - start_time
            status = response.status
            logger.info(f"Request {request_id}: status={status}, time={response_time:.3f}s")
            return {
                "request_id": request_id,
                "status": status,
                "response_time": response_time
            }
    except Exception as e:
        logger.error(f"Request {request_id} failed: {str(e)}")
        return {
            "request_id": request_id,
            "status": "error",
            "error": str(e)
        }

async def run_stress_test(requests_count, concurrency):
    """Run a stress test with specified number of requests and concurrency level"""
    conn = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=conn) as session:
        tasks = []
        for i in range(requests_count):
            tasks.append(make_request(session, i))
        
        return await asyncio.gather(*tasks)

def handler(event, context):
    # Get test parameters from the event or use defaults
    requests_count = event.get('requests_count', 100)  # Total number of requests
    concurrency = event.get('concurrency', 10)        # Max concurrent requests
    
    logger.info(f"Starting stress test with {requests_count} requests, concurrency {concurrency}")
    
    # Use ThreadPoolExecutor to run asyncio event loop in Lambda
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            lambda: asyncio.get_event_loop().run_until_complete(
                run_stress_test(requests_count, concurrency)
            )
        )
        results = future.result()
    
    # Calculate metrics
    successful = sum(1 for r in results if isinstance(r.get('status'), int) and 200 <= r.get('status') < 300)
    failed = sum(1 for r in results if isinstance(r.get('status'), int) and r.get('status') >= 400)
    errors = sum(1 for r in results if r.get('status') == "error")
    
    response_times = [r.get('response_time') for r in results if r.get('response_time')]
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
    
    metrics = {
        "total_requests": requests_count,
        "successful_requests": successful,
        "failed_requests": failed,
        "errors": errors,
        "average_response_time": avg_response_time
    }
    
    logger.info(f"Stress test completed. Metrics: {json.dumps(metrics)}")
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'metrics': metrics,
            'message': 'Stress test completed successfully'
        })
    }