
import sys
sys.path.append(".")

from src.llm_client import ask

print(ask("Write a one line story about a cat"))