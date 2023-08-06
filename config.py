from dataclasses import dataclass
from functools import lru_cache
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    OPENAI_API_KEY: str
    PROMPTS_FILE: str
    OPENAI_MODEL: str
    DATA_DIR: str


@lru_cache()
def get_config() -> Settings:
    return Settings(
        OPENAI_API_KEY=os.getenv("OPENAI_API_KEY"),
        OPENAI_MODEL=os.getenv("OPENAI_MODEL", "gpt-4"),
        PROMPTS_FILE=os.getenv("PROMPTS_FILE", "prompts.json"),
        DATA_DIR=os.getenv("DATA_PATH", "./data"),
    )
