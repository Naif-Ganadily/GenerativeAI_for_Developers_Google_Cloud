[Link to the Notebook](notebooks\youtube_analysis.ipynb) <br>
[Link to the Google Cloud Lab]()
# Notebook review

## Introduction
This document summarizes my learnings and key steps taken while going through the lab notebook focused on utilizing Vertex AI LLM for various natural language processing tasks.

## Initial Setup
- **Packages Installation**: Started by installing essential Python packages which include `chromadb`, `gradio`, `pytube`, `youtube-transcript-api`, `pydantic`, and `langchain`, along with Google Cloud packages necessary for the lab.
  
  ```bash
  !pip install --upgrade --user google-cloud-aiplatform>=1.29.0 google-cloud-storage langchain pytube youtube-transcript-api chromadb gradio pydantic

## Vertex AI LLM Initialization
- Model Import: Imported text-bison@001 from Vertex AI, a model designed for a range of natural language tasks.

```python
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import VertexAI

llm = VertexAI(
    model_name="text-bison@001",
    max_output_tokens=256,
    temperature=0.1,
    top_p=0.8,
    top_k=40,
    verbose=True,
)
```

- Utilized for tasks including summarization, question answering, sentiment analysis, entity extraction, and classification.

## Data Processing and Analysis
- **Embeddings Conversion:** Used VertexAIEmbeddings from the LangChain library to convert video chunks into embeddings, facilitating the processing of video content into manageable text chunks.

```python
from langchain.embeddings import VertexAIEmbeddings

EMBEDDING_QPM = 100
EMBEDDING_NUM_BATCH = 5

embeddings = VertexAIEmbeddings(
    requests_per_minute=EMBEDDING_QPM,
    num_instances_per_batch=EMBEDDING_NUM_BATCH,
)
```

- **Video Content Processing:** Loaded video content using the YouTube loader, split it into text documents for further processing, and stored the documents in ChromaDB for retrieval and semantic search.

```python
loader = YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=XX2XpqklUrE", add_video_info=True)
result = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
docs = text_splitter.split_documents(result)
print(f"# of documents = {len(docs)}")
```


## Question Answering with Retrieval QA Chain
- **Retrieval QA Setup:** Configured a Retrieval QA Chain using ChromaDB and the Vertex AI model to perform question answering by retrieving relevant document embeddings and providing them as context to the model.

```python
db = Chroma.from_documents(docs, embeddings)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
```


- **Stuffing Methodology:** Learned about the stuffing method where the context is "stuffed" into a prompt to help the model process all relevant information and generate meaningful responses.

```python
def sm_ask(question, print_results=True):
  video_subset = qa({"query": question})
  context = video_subset
  prompt = f"""
  Question:
  {question}
  Text:
  {context}
  
  Answer:
  """
  parameters = {
    "temperature": 0.1,
    "max_output_tokens": 256,
    "top_p": 0.8,
    "top_k": 40
  }
  response = llm.predict(prompt, **parameters)
  
  return {
    "answer": response
  }
```

## Conclusion and Next Steps
- Successfully completed the lab, gaining hands-on experience with LangChain applications for processing and analyzing video content through natural language tasks.
- Explored the documentation for Generative AI on Vertex AI and the Google Cloud Tech YouTube channel for further learning.
- Considered pursuing Google Cloud training and certification to deepen understanding of cloud technologies and applications in natural language processing.



