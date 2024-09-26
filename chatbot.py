import requests
import time
import aiohttp
import asyncio

async def get_response(user_input):
    ollama_url = 'http://127.0.0.1:11434/v1/chat/completions'
    headers = {
        'Content-Type': 'application/json',
    }
    data = {
        'model': 'llama2',
        'messages': [
            {'role': 'user', 'content': user_input}
        ]
    }

    start_time = time.time()

    async with aiohttp.ClientSession() as session:
        async with session.post(ollama_url, headers=headers, json=data) as response:
            elapsed_time = time.time() - start_time
            if response.status == 200:
                response_data = await response.json()
                bot_response = response_data['choices'][0]['message']['content']
            else:
                bot_response = "Error: Could not fetch a response."

    bot_response += f"\n\nResponse Time: {elapsed_time:.2f} seconds"
    return bot_response

async def main():
    user_input = input("Enter your input: ")
    response = await get_response(user_input)
    print(response)

if __name__ == '__main__':
    asyncio.run(main())
