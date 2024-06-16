from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import pickle
import json
from langchain.vectorstores import Chroma
import ast
from collections import defaultdict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


class LLM:
    def __init__(self, prompt_template) -> None:
        self.embedding = HuggingFaceInstructEmbeddings(
            model_name='hkunlp/instructor-large'
        )
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=50
        )
        self.llm = LlamaCpp(
            model_path="D:/llms/solar-10.7b-instruct-v1.0.Q8_0.gguf",
            temperature=0.1,
            n_gpu_layers= 30,
            max_tokens = 1024,
            n_batch = 4096,
            n_ctx = 8092,
            stop = ['\n\n\n\n\n', "}", "\n \n \n "],
            callback_manager=callback_manager,
            verbose=True,
            grammar_path='./quallm/llm/json.gbnf'
        )
        self.prompt_template = prompt_template

    def __call__(self, question, data, corpus, rag=True, n_runs=1) -> dict:
        if rag:
            all_splits = self.text_splitter.split_text(data)
            vectorstore = Chroma.from_texts(
                texts=all_splits, embedding=self.embedding
            )

            prompt_template = self.prompt_template.partial(corpus = corpus)
            qa_chain = RetrievalQA.from_chain_type(
                self.llm,
                retriever=vectorstore.as_retriever(search_kwargs={"k": 6}),
                chain_type_kwargs={"prompt": prompt_template},
            )
            
            for _ in range(n_runs):
                llm_response = qa_chain({"query": question})['result']

            vectorstore.delete_collection()
        else:
            prompt = self.prompt_template.format(
                corpus=corpus, context=data, question=question
            )
            llm_response = self.llm(prompt)

        return llm_response + "}"
