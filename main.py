from fastapi import FastAPI
import google.generativeai as genai
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from typing import Any, List, Mapping, Optional

import os
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

app = FastAPI()

# model = genai.GenerativeModel('gemini-pro')

class GeminiProLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "gemini-pro"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        
        gemini_pro_model = genai.GenerativeModel('gemini-pro')

        
        model_response = gemini_pro_model.generate_content(
            prompt, 
            generation_config={"temperature": 0.1}
        )
        text_content = model_response.candidates[0].content.parts[0].text
        return text_content

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model_id": "gemini-pro", "temperature": 0.1}

def load_chain():
    llm = GeminiProLLM()
    memory = ConversationBufferMemory()
    chain = ConversationChain(llm=llm, memory=memory)
    return chain

chatchain = load_chain()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/chatGemini")
async def chat_gemini(qurry: dict):
    response = chatchain(qurry["user"])
    return response
