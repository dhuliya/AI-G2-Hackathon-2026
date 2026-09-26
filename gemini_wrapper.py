import os
from dotenv import load_dotenv
import google.generativeai as genai
from deepeval.models.base_model import DeepEvalBaseLLM

load_dotenv()

class GoogleGemini(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        self.model_name = model_name
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text

    async def a_generate(self, prompt: str) -> str:
        response = await self.model.generate_content_async(prompt)
        return response.text

    def get_model_name(self):
        return self.model_name