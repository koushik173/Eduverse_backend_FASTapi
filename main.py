from fastapi import FastAPI
import google.generativeai as genai
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from typing import Any, List, Mapping, Optional
from fastapi.middleware.cors import CORSMiddleware
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory

from langchain_community.document_loaders import YoutubeLoader


from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.callbacks import get_openai_callback
import os
import json
import pandas as pd
import traceback
from dotenv import load_dotenv
import PyPDF2

os.environ["OPENAI_API_KEY"] = "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"
llm=ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0.7)



load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_indexs")
    return {"done": "ok"}


async def chatt(texttt):
    response = chatchain(texttt)
    return response['response']


# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")

def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="question")
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt, memory=memory)

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_indexs", embeddings)
    docs = new_db.similarity_search(user_question)
    

    chain = get_conversational_chain()
    response = chain(
        {"input_documents":docs, "question": user_question, "history": "history"}
        , return_only_outputs=True)

    return response


loader = YoutubeLoader.from_youtube_url(
    "https://www.youtube.com/watch?v=dH54GOP7PVY",
    add_video_info=True,
    language=["en", "id"],
    translation="en",
)

def genMCQ(text, number, subject, tone):
    RESPONSE_JSON = {'1': {'no': '1',
        'mcq': 'multiple choice questions',
        'options': {'a': 'choice here',
        'b': 'choice here',
        'c': 'choice here',
        'd': 'choice here'},
        'correct': 'correct answer'},
        '2': {'no': '2',
        'mcq': 'multiple choice questions',
        'options': {'a': 'choice here',
        'b': 'choice here',
        'c': 'choice here',
        'd': 'choice here'},
        'correct': 'correct answer'},
        '3': {'no': '3',
        'mcq': 'multiple choice questions',
        'options': {'a': 'choice here',
        'b': 'choice here',
        'c': 'choice here',
        'd': 'choice here'},
        'correct': 'correct answer'}}
    
    TEMPLATE="""
    Text:{text}
    You are an expert MCQ maker. Given the above text, it is your job to \
    create a quiz  of {number} multiple choice questions for {subject} students in {tone} tone. 
    Make sure the questions are not repeated and check all the questions to be conforming the text as well.
    Make sure to format your response like  RESPONSE_JSON below  and use it as a guide. \
    Ensure to make {number} MCQs
    ### RESPONSE_JSON
    {RESPONSE_JSON}

    """
    quiz_generation_prompt=PromptTemplate(
        input_variables=["text","number","subject","tone","RESPONSE_JSON"],
        template=TEMPLATE
    )
    quiz_chain=LLMChain(llm=llm,prompt=quiz_generation_prompt,output_key="quiz")

    TEMPLATE2="""
    You are an expert english grammarian and writer. Given a Multiple Choice Quiz for {subject} students.\
    You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for complexity analysis. 
    if the quiz is not at per with the cognitive and analytical abilities of the students,\
    update the quiz questions which needs to be changed and change the tone such that it perfectly fits the student abilities
    Quiz_MCQs:
    {quiz}

    Check from an expert English Writer of the above quiz:
    """

    quiz_evaluation_prompt=PromptTemplate(input_variables=["subject", "quiz"], template=TEMPLATE2)
    review_chain=LLMChain(llm=llm,prompt=quiz_evaluation_prompt,output_key="review")
    generate_evaluate_chain=SequentialChain(chains=[quiz_chain, review_chain], input_variables=["text", "number", "subject", "tone", "RESPONSE_JSON"],output_variables=["quiz", "review"])


    with  get_openai_callback() as cb:
        response=generate_evaluate_chain(
            {

            "text": text,
            "number": number,
            "subject": subject,
            "tone": tone,
            "RESPONSE_JSON": json.dumps(RESPONSE_JSON)

            }
        )

    quiz=response.get("quiz")
    final_quiz=json.loads(quiz)
    return final_quiz


@app.get("/")
def read_root():
    return {"Hello": "Gemini"}
 
 
@app.post("/chatGemini")
async def chat_gemini(qurry: dict):

    res = await chatt(qurry["user"])
    print(res)
    return res


@app.post("/chatPDFgemini")
async def chat_pdf_gemini(qurry: dict):
    # text_chunks = get_text_chunks(text)
    # get_vector_store(text_chunks) 
    response = user_input(qurry["user"])
    return response

@app.post("/pdfTextLoaded")
async def pdf_text_loaded(qurry: dict):
    res = get_text_chunks(qurry["usertext"])
    return {"success": True}

@app.post("/chat_Youtubegemini")
async def chat_youtube_gemini(qurry: dict):
    response = user_input(qurry["user"])
    return response

@app.post("/load_YoutubeText")
async def load_youtube_text(link: dict):
    loader = YoutubeLoader.from_youtube_url(
        link["url"],
        add_video_info=True,
        language=["en", "id"],
        translation="en",
    )
    text = loader.load()
    page_content = text[0].page_content
    get_text_chunks(page_content)
    # text_chunks = get_text_chunks(text)
    # get_vector_store(text_chunks)
    return {"success": True}

@app.post("/mcq_Gendoc")
async def mcq_genDoc(doc: dict):
    text = doc['text']
    number = doc['number']
    subject = doc['subject']
    tone = doc['tone']
    mcq = genMCQ(text,number,subject,tone)

    return mcq
 
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    