from dotenv import load_dotenv
from langfuse import get_client

load_dotenv()

client = get_client()

print("auth ok:", client.auth_check())
