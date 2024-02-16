import json
from llama_index import (
    KeywordTableIndex,
    SimpleDirectoryReader,
    LLMPredictor,
    ServiceContext,
)
from llama_index.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index import StorageContext,load_index_from_storage
import os
from llama_index import Prompt
from llama_index.prompts import PromptTemplate
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.query_engine import CustomQueryEngine
from llama_index.retrievers import BaseRetriever
from llama_index.response_synthesizers import get_response_synthesizer, BaseSynthesizer
from llama_index.retrievers import VectorIndexRetriever
from deep_translator import GoogleTranslator
from rouge_score import rouge_scorer
from langdetect import detect
import nltk.translate.meteor_score as meteor
from nltk.translate.bleu_score import corpus_bleu


import nltk
nltk.download('wordnet')






os.environ['OPENAI_API_KEY'] = "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"







def extract_news_desc():
    links = []
    headlines = []

    # Assuming your JSON data is stored in a file named 'data.json'
    with open('/content/News_Category_Dataset_v3.json') as f:
        for line in f:
            # Load each JSON object separately
            record = json.loads(line)

            # Extract link and headline from the record
            links.append(record['link'])
            headlines.append(record['headline'])

    # Print the extracted lists
    print("Links:", links[0:50])
    print("Headlines:", headlines[0:50])


def create_index():

    llm = OpenAI(temperature=0.1, model="gpt-4")
    service_context = ServiceContext.from_defaults(llm=llm)

    documents = SimpleDirectoryReader("/data").load_data()
    index = VectorStoreIndex.from_documents(documents,service_context=service_context,show_progress=True)  

    index.storage_context.persist(persist_dir='/content/Emed2') 

def processing_without_Custom_QE():

    storage_context = StorageContext.from_defaults(persist_dir='/content/drive/MyDrive/Emed2')
    index = load_index_from_storage(storage_context)

    query_engine = index.as_query_engine(llm=ChatOpenAI())
    #TESTCASE 1--
    response = query_engine.query("give me article about Over 4 Million Americans Roll Up Sleeves For Omicron-Targeted COVID Boosters in 500 words")
    # response = llm.complete(prompt)
    # print(response.text)
    print(response)

    #TESTCASE 2--
    response = query_engine.query("give me 10 most useful keywords from the summary of the article Over 4 Million Americans Roll Up Sleeves For Omicron-Targeted COVID Boosters in points")
    # response = llm.complete(prompt)
    # print(response.text)
    print(response)

    #TESTCASE 3--
    response = query_engine.query('''give me realted articles from these keywords :-1. Americans
    2. Omicron-Targeted COVID Boosters
    3. COVID-19 booster shot
    4. Centers for Disease Control and Prevention
    5. President Joe Biden
    6. pandemic
    7. White House
    8. vaccine
    9. flu shot
    10. infectious disease epidemiologist
    other than Over 4 Million Americans Roll Up Sleeves For Omicron-Targeted COVID Boosters''')
    # response = llm.complete(prompt)
    # print(response.text)
    print(response)

    #TESTCASE 4--
    response = query_engine.query("give me the summary of the article Why Getting a Flu Shot is Important During the COVID-19 Pandemic in points")
    print(response)

    #TESTCASE 5--
    template1 = ('''
    This is a example of how you will summarize this news-article {query_str}  in 60 words according to its content and staying focused to its content and description
    Summarization points-
    (Example – User Input query – India Interim Budget Details for 2024
    Output Expectation -
    Summarization Points -
    Finance Minister Nirmala Sitharaman stressed on 5 ‘Disha Nirdashak’ baatein: Social justice as an
    effective governance model; Focus on the poor, youth, women, and the Annadata (farmers); Focus on
    infrastructure; Use of technology to improve productivity and High-power committee for challenges
    arising from demographic challenges.)
    ''')
    #qa_template = Prompt(template1)
    qa_template = PromptTemplate(template1)
    query='''France lost 35 citizens, Thailand 33, US 31, Ukraine 21, Russia 19, UK 12, Nepal 10, Germany 10,'''
    qa_template.format(query_str=query)
    query_engine = index.as_query_engine(text_qa_template=qa_template)
    #query_engine = index.as_query_engine()
    response = query_engine.query(query)
    # response = llm.complete(prompt)
    # print(response.text)
    print(response)

def processing_with_Custom_QE(query,index,n_words):
    # storage_context = StorageContext.from_defaults(persist_dir='D:/Resc/client/data/Emed2')
    # index = load_index_from_storage(storage_context)

    qa_prompt = PromptTemplate(
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    """
    You are a excellent news article summarizer that firstly summarizes the article in {n_words} from the trained embeddings in 6 points then it extracts meaningful topic article realted keywords from trained embeddings and then give the links of news article  related to those keywords from the embeddings in a template given below
        Summarization points:\n
        1. [Summarized Point 1]\n
        2. [Summarized Point 2]\n
        3. [Summarized Point 3]\n
        4. [Summarized Point 4]\n
        5. [Summarized Point 5]\n
        6. [Summarized Point 6]\n

        Related Links:\n
        [Link 1 :]\n
        [Link 2 :]\n

        Keywords:\n
        The extracted keywords from the summarized text which is generated by the model are [Keyword 1], [Keyword 2], [Keyword 3], [Keyword 4], [Keyword 5]\n

        Your model should suggest articles related to two links provided above:\n
        1. [keyword related article headline(without link)]\n
        2. [keyword related article headline(without link)]\n

    """
    )
    qa_prompt1 = PromptTemplate(
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    """give a summary of {query_str} and also related keywords in points"""
    )

    class RAGStringQueryEngine(CustomQueryEngine):
        """RAG String Query Engine."""

        retriever: BaseRetriever
        response_synthesizer: BaseSynthesizer
        llm: OpenAI
        qa_prompt: PromptTemplate

        def custom_query(self, query_str: str):
            nodes = self.retriever.retrieve(query_str)

            context_str = "\n\n".join([n.node.get_content() for n in nodes])
            response = self.llm.complete(
                qa_prompt.format(context_str=context_str, query_str=query_str, n_words = n_words)
            )

            return str(response)
    
    retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=1,
    )

    # configure response synthesizer
    response_synthesizer = get_response_synthesizer(
        response_mode="tree_summarize",
    )

    llm = OpenAI(model="gpt-3.5-turbo")

    query_engine = RAGStringQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        llm=llm,
        qa_prompt=qa_prompt,
    )
    translated = GoogleTranslator(source='auto', target='en').translate(query)
    language = detect(query)
    trans = GoogleTranslator(source='auto', target='kn').translate(translated)
    trans1 = GoogleTranslator(source='auto', target='en').translate(trans)
    print(trans1)

    response = query_engine.query(translated)
    resp = str(response)


































    human_written_article = """“Hurricane Fiona has severely impacted electrical infrastructure and generation facilities throughout the island. We want to make it very clear that efforts to restore and reenergize continue and are being affected by severe flooding, impassable roads, downed trees, deteriorating equipment, and downed lines,” said Luma, the company that operates power transmission and distribution.

Neighbors work to recover their belongings from the flooding caused by Hurricane Fiona in the Los Sotos neighborhood of Higüey, Dominican Republic, on Sept. 20, 2022.
Neighbors work to recover their belongings from the flooding caused by Hurricane Fiona in the Los Sotos neighborhood of Higüey, Dominican Republic, on Sept. 20, 2022.AP PHOTO/RICARDO HERNANDEZ
The hum of generators could be heard across the island as people became increasingly exasperated, with some still trying to recover from Hurricane Maria, which hit as a Category 4 storm five years ago, killing an estimated 2,975 people in its aftermath.

Luis Noguera, who was helping clear a landslide in the central mountain town of Cayey, said Maria left him without power for a year.

“We paid an electrician out of our own pocket to connect us,” he recalled, adding that he doesn’t think the government will be of much help again after Fiona.

Long lines were reported at several gas stations across Puerto Rico, and some pulled off a main highway to collect water from a stream.

“We thought we had a bad experience with Maria, but this was worse,” said Gerardo Rodríguez, who lives in the southern coastal town of Salinas.

Parts of the island had received more than 25 inches (64 centimeters) of rain and more had fallen on Tuesday.

By late Tuesday, authorities said they had restored power to nearly 300,000 of the island’s 1.47 million customers, while water service was cut to more than 760,000 customers — two thirds of the total on the island.

The head of the Federal Emergency Management Agency traveled to Puerto Rico on Tuesday as the agency announced it was sending hundreds of additional personnel to boost local response efforts.

Meanwhile, the U.S. Department of Health and Human Services declared a public health emergency on the island and deployed a couple of teams to the U.S. territory.”"""
    # response_wiz = query_engine_llama.query(query)
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = scorer.score(human_written_article,resp )
    references = [[human_written_article]]
    candidates = [resp]
    bleu_score = corpus_bleu(references, candidates)

    #Fiona Threatens To Become Category 4 Storm Headed To Bermuda
    # Print scores
    print("BLEU score:", bleu_score)
    # print("METEOR score:", meteor_score)
    print(scores)
    print(language)
    resp = GoogleTranslator(source='auto', target=language).translate(resp)
    return str(resp)
    # return query