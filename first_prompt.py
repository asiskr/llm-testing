import google.generativeai as genai
from dotenv import load_dotenv
import os
import warnings

load_dotenv()
warnings.filterwarnings("ignore")

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


model = genai.GenerativeModel("gemini-3.6-flash", generation_config={"temperature": 1})

response = model.generate_content("Write a one-line story about a cat")

print(response.text)