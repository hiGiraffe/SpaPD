import os
import asyncio
import aiohttp
from quart import Quart, make_response, request
import time

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

app = Quart(__name__)


async def forward_request(url, data):
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
        }
        async with session.post(url=url, json=data,
                                headers=headers) as response:
            if response.status == 200:
                # if response.headers.get('Transfer-Encoding') == 'chunked':
                if True:
                    async for chunk_bytes in response.content.iter_chunked(
                            1024):
                        yield chunk_bytes
                else:
                    content = await response.read()
                    yield content


async def handle_request(original_request_data):
    try:
        prefill_request = original_request_data.copy()
        # change max_tokens = 1 to let it only do prefill
        prefill_request['max_tokens'] = 1
        # finish prefill
        async for _ in forward_request('http://localhost:8100/v1/completions',
                                       prefill_request):
            continue
        # return decode
        generator = forward_request('http://localhost:8200/v1/completions',
                                    original_request_data)
        
        # 收集 8200 的输出
        output_8200 = ""
        async for chunk in generator:
            output_8200 += chunk.decode('utf-8')
            
        # print(output_8200)
        # response = await make_response(generator)
        # response.timeout = None

        return output_8200

    except Exception as e:
        import sys
        import traceback
        exc_info = sys.exc_info()
        print("Error occurred in disagg prefill proxy server")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))

async def main():
    input1 = {
        'model': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'prompt': ' reat base for exploring the nearby islands, including Aegina, Poros, and Hydra. Take a ferry or a boat to explore these beautiful islands and enjoy the stunni',
        'max_tokens': 300,
        'temperature': 0
    }
    input2 = {
        'model': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'prompt': 'oy the stunni',
        'max_tokens': 300,
        'temperature': 0
    }

    # 并行执行 handle_request
    results = await asyncio.gather(
        handle_request(input1),
        handle_request(input2)
    )

    sleep_time = 0.5
    await asyncio.sleep(sleep_time)
    
    await handle_request(input1)
    
"""
实验1：
输入shareGPT的每一请求，每次获取到结果后，睡眠1s后继续输入，记录各个部分的具体时间
"""
async def motivation1():
    model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    max_tokens = 2000
    temperature = 0
    
    inputs = ["Summarize the main ideas of Jeff Walker's Product Launch Formula into bullet points as it pertains to a growth marketing agency implementing these strategies and tactics for their clients...", "Summarize the main ideas of Brendon Burchard's Experts Academy into bullet points as it pertains to a growth marketing agency implementing these strategies and tactics for their clients...", 'What are the mental triggers in Jeff Walker\'s Product Launch Formula and "Launch" book?', 'Write a summary of why scarcity and urgency are the strongest mental triggers and have been the driving force behind many of our best performing campaigns over the last 8 years.', "Summarize Russell Brunson's Perfect Webinar Script...", 'Summarize the 6 human needs as Tony Robbins explains...']
    old_context = ""
    start_time = time.time()
    for new_context in inputs:
        input = {
            'model': model_name,
            'prompt': old_context+new_context,
            'max_tokens': max_tokens,
            'temperature': temperature
        }
        old_context = await handle_request(input)
    
    
    # # 并行执行 handle_request
    # results = await asyncio.gather(
    #     handle_request(input1),
    #     handle_request(input2)
    # )

if __name__ == '__main__':    
    # 运行主异步函数
    asyncio.run(main())
    
    # 运行motivation1
    # asyncio.run(motivation1())