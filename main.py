from fastapi import FastAPI
import google.generativeai as genai
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from typing import Any, List, Mapping, Optional

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



GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

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


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
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
    
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)
    

    chain = get_conversational_chain()
    response = chain(
        {"input_documents":docs, "question": user_question, "history": "history"}
        , return_only_outputs=True)

    return response

text = """How do emotions influence decision-making processes in individuals, and to what extent do external factors contribute to shaping emotional responses?
Emotions play a pivotal role in human decision-making, acting as powerful influencers that guide choices and actions. The intricate interplay between emotions and decision-making processes is a fascinating area of study that delves into the depths of psychology, neuroscience, and behavioral economics. Understanding the nuances of this relationship requires exploration into the mechanisms through which emotions arise, how they are processed, and the extent to which external factors contribute to shaping emotional responses.
At the core of emotional influence on decision-making lies the limbic system, a complex network of brain structures responsible for processing emotions. The amygdala, in particular, plays a crucial role in the emotional responses associated with decision-making. This almond-shaped structure is responsible for processing and interpreting emotional stimuli, triggering physiological responses that prepare the body for action. Consequently, the emotional signals generated by the amygdala can significantly impact the decision-making process, either enhancing or impairing cognitive functions.
The impact of emotions on decision-making is not limited to the neural level; it extends to the psychological and behavioral dimensions as well. Emotions can serve as valuable guides, providing individuals with insights into their preferences, values, and potential risks associated with different choices. Positive emotions, such as joy or excitement, may lead to more risk-taking behavior, while negative emotions, like fear or sadness, can encourage cautious decision-making.
However, the relationship between emotions and decision-making is not a one-way street. External factors, such as social, cultural, and environmental influences, also contribute significantly to shaping emotional responses. Social context, for instance, can modulate emotional experiences, with individuals often conforming to societal norms and expectations. Cultural differences further highlight the variability in emotional expression and its impact on decision-making across diverse populations.
Moreover, environmental stimuli can trigger emotional responses that, in turn, influence decision-making. The concept of environmental priming suggests that the physical environment can activate specific emotional states, subsequently influencing behavior. For instance, a well-lit and aesthetically pleasing environment may promote positive emotions, potentially leading to more optimistic decision-making.
The influence of external factors on emotional responses raises questions about the autonomy of individual decision-making. To what extent are decisions truly reflective of personal preferences and rational considerations, and how much do external influences shape the emotional landscape that guides these decisions? Unraveling this intricate web of interactions requires a comprehensive examination of both internal and external contributors to the decision-making process.
Additionally, the role of individual differences in emotional regulation and perception cannot be overlooked. People vary in their ability to regulate emotions, with factors such as personality, upbringing, and past experiences shaping emotional resilience. Understanding how these individual differences intersect with external influences can provide valuable insights into the variability observed in decision-making across diverse populations.
In conclusion, the relationship between emotions and decision-making is a multifaceted and dynamic interplay influenced by neural, psychological, and external factors. Exploring the intricate mechanisms through which emotions shape decisions and understanding the extent to which external influences contribute to emotional responses are essential for gaining a holistic perspective on human behavior. This exploration not only enhances our understanding of decision-making processes but also has implications for fields ranging from psychology and neuroscience to economics and public policy."""



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
    response = chatchain(qurry["user"])
    return response

@app.post("/chatPDFgemini")
async def chat_pdf_gemini(qurry: dict):
    text_chunks = get_text_chunks(text)
    get_vector_store(text_chunks)

    response = user_input(qurry["user"])
    return response

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
    # text_chunks = get_text_chunks(text)
    # get_vector_store(text_chunks)

    return {"page_content":page_content}


@app.post("/mcq_Gendoc")
async def mcq_genDoc(doc: dict):
    text = doc['text']
    number = doc['number']
    subject = doc['subject']
    tone = doc['tone']
    mcq = genMCQ(text,number,subject,tone)

    return mcq


