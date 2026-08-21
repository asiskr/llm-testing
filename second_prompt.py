import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


model = genai.GenerativeModel("gemini-3.6-flash", generation_config={"temperature": 0})

response = model.generate_content("Write a one-line story about a cat")

print(response.text)