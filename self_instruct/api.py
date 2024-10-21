import os
import time
import traceback
from datetime import datetime

import openai
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


def get_gen(text: str, system: str = None) -> str:
    messages = [
        {
            "role": "user", 
            "content": text
        },
    ]
    if system:
        messages.insert(
            0,
        {
            "role": "system", 
            "content": system
        })
    # print(messages)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        stream=False,
    )
    resp = ""
    try:
        resp = response.choices[0].message.content
    except Exception as e:
        traceback.print_exc()
    return resp


def make_requests(
        engine, prompts, max_tokens, temperature, top_p,
        frequency_penalty, presence_penalty, stop_sequences, logprobs, n, best_of, retries=3,
        api_key=api_key
    ):
    response = None
    target_length = max_tokens
    if api_key is not None:
        openai.api_key = api_key
        
    retry_cnt = 0
    backoff_time = 30
    while retry_cnt <= retries:
        try:
            response = openai.completions.create(
                model=engine,
                prompt=prompts,
                max_tokens=target_length,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop_sequences,
                logprobs=logprobs,
                n=n,
                best_of=best_of
            ).model_dump()
            break
        except Exception as e:
            print(f"OpenAIError: {e}.")
            if "Please reduce your prompt" in str(e):
                target_length = int(target_length * 0.8)
                print(f"Reducing target length to {target_length}, retrying...")
            else:
                print(f"Retrying in {backoff_time} seconds...")
                time.sleep(backoff_time)
                backoff_time *= 1.5
            retries += 1
            
    if isinstance(prompts, list):
        # print("response: ", response)
        results = []
        for j, prompt in enumerate(prompts):
            data = {
                "prompt": prompt,
                "response": {"choices": response["choices"][j * n: (j + 1) * n]} if response else None,
                "create_at": str(datetime.now())
            }
            results.append(data)
        return results
    else:
        data = {
            "prompt": prompts,
            "response": response,
            "create_at": str(datetime.now())
        }
        return [data]
    

if __name__ == "__main__":
    system = "you name is liliha"
    text = "what is your name"
    result = get_gen(text, system)
    print(result)
    
    result = make_requests(
        engine="davinci-002",
        prompts=["hi", "you are"],
        max_tokens=32,
        temperature=0.7,
        top_p=0.5,
        frequency_penalty=0,
        presence_penalty=2,
        stop_sequences=["\n\n", "\n16", "16.", "16 ."],
        logprobs=1,
        n=1,
        best_of=1,
    )
    print(result)