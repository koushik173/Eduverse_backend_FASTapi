from fastapi import FastAPI
import google.generativeai as genai
import os
from fastapi.middleware.cors import CORSMiddleware

import pathlib
import textwrap


from IPython.display import display
from IPython.display import Markdown

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))


from fastapi import FastAPI
import google.generativeai as genai
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from typing import Any, List, Mapping, Optional


# GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key="AIzaSyDGrJ6j8XUMVnn-yQejVtnNX2LdErW8qhw")



app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# async def chatt(text):
#     # model = genai.GenerativeModel('gemini-pro')
#     # response = await model.generate_content("What is the meaning of life Give ans in 50 words")

#     return {"Hello": "test"}




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

async def chatt(texttt):
    # model = genai.GenerativeModel('gemini-pro')
    # response = model.generate_content(texttt)
    print("okkkk", texttt)

    response = chatchain(texttt)
    print("okkkk")
    print(response)
    # print("okkkk", response['response'])

    return {"Hello": response['response']}

@app.get("/")
def read_root():
    return {"Hello": "Gemini"}

@app.post("/chatGemini")
async def chat_gemini(qurry: dict):
    res = await chatt(qurry["user"])
    return res

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)





