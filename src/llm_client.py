import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL_NAME = "openai/gpt-oss-20b"

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def ask(prompt, temperature=0):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content

def chat(messages, temprature=0):
    response = client.chat.completions.create(
        model= MODEL_NAME,
        messages= messages,
        temperature=temprature,
    )
    return response.choices[0].message.content