---
title: "LangChain: A Powerful Tool for Local LLM Execution"
date: 2023-09-30
tags: [LLM, Local LLM, Generative AI, Langchain, Chroma]
excerpt: "Working with Local LLMs"
header:
  overlay_image: "\images\local-llm\home-page.jpg"
  caption: "Photo by <a href="https://unsplash.com/@omilaev?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Igor Omilaev</a> on <a href="https://unsplash.com/photos/a-computer-chip-with-the-letter-a-on-top-of-it-eGGFZ5X2LnA?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Unsplash</a>"  
mathjax: "true"
---

## Introduction to Langchain and Local LLMs

### Langchain

LangChain is a framework for developing applications powered by language models. It enables applications that:

* **Are context-aware**: connect a language model to sources of context (prompt instructions, few shot examples, content to ground its response in, etc.)
* **Reason**: rely on a language model to reason (about how to answer based on provided context, what actions to take, etc.)

The main value props of LangChain are:

* Components: abstractions for working with language models, along with a collection of implementations for each abstraction. Components are modular and easy to use, whether you are using the rest of the LangChain framework or not
* Off-the-shelf chains: a structured assembly of components for accomplishing specific higher-level tasks

Off-the-shelf chains make it easy to get started. For complex applications, components make it easy to customize existing chains and build new ones.

### Local LLMs

Local LLMs, or local large language models, are LLMs that can be run on your own computer or server. This means that you don’t need to rely on a cloud service to use them, which can offer a number of advantages, including:

* **Data privacy and security**: When you run a local LLM, your data never leaves your device. This can be important for sensitive data, such as healthcare records or financial data.
* **Offline availability**: Local LLMs can be used offline, which means that you can use them even if you don’t have an internet connection. This can be useful for working on projects in remote areas or for applications that need to be available all the time.
* **Customization**: Local LLMs can be fine-tuned for specific tasks or domains. This can make them more accurate and efficient for the tasks that you need them to do.

LLMs can be run on a variety of hardware platforms, including CPUs and GPUs. However, it is important to note that local LLMs can be very computationally expensive to run, so you may need a powerful computer to use them effectively.

To run a local LLM, you will need to install the necessary software and download the model files. Once you have done this, you can start the model and use it to generate text, translate languages, answer questions, and perform other tasks.

Here are some examples of how local LLMs can be used:

* Generating creative content: Local LLMs can be used to generate creative content, such as poems, stories, and code. This can be useful for writers, artists, and programmers alike.
* Translating languages: Local LLMs can be used to translate languages more accurately and efficiently than traditional machine translation systems. This can be useful for businesses and individuals who need to communicate with people who speak other languages.
* Answering questions: Local LLMs can be used to answer questions in a comprehensive and informative way. This can be useful for students, researchers, and anyone else who needs to learn more about a particular topic.

In this article, we will run a local LLM to perform the “Answering Question” task.

## Setting Up the Environment

### Installing Required Python Packages

Before you can start running a Local LLM using Langchain, you’ll need to ensure that your development environment is properly configured. Let’s start by installing the required libraries.

```shell
pip install transformers langchain tiktoken chromadb pypdf InstructorEmbedding accelerate bitsandbytes sentence-transformers
```

## Configuring Langchain for Local LLMs

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
```

### Initializing the Sequence-to-Sequence Model and Tokenizer

In this article, we will explore the process of running a local Language Model (LLM) on a local system, and for demonstration purposes, we will be utilizing the “FLAN-T5” model.

```python
# Pass the directory path where the model is stored on your system
model_name = "./google/flan-t5-large"

# Initialize a tokenizer for the specified model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize a model for sequence-to-sequence tasks using the specified pretrained model
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
```

### Creating a Text-to-Text Generation Pipeline

```python
# Create a pipeline for text-to-text generation using a specified model and tokenizer
pipe = pipeline(
    "text2text-generation",  # Specify the task as text-to-text generation
    model=model,              # Use the previously initialized model
    tokenizer=tokenizer,      # Use the previously initialized tokenizer
    max_length=512,           # Set the maximum length for generated text to 512 tokens
    temperature=0,            # Set the temperature parameter for controlling randomness (0 means deterministic)
    top_p=0.95,               # Set the top_p parameter for controlling the nucleus sampling (higher values make output more focused)
    repetition_penalty=1.15   # Set the repetition_penalty to control the likelihood of repeated words or phrases
)
```

### Implementing a Local Language Model for Text Generation

```python
# Create a Hugging Face pipeline for local language model (LLM) using the 'pipe' pipeline
local_llm = HuggingFacePipeline(pipeline=pipe)

# Generate text by providing an input prompt to the LLM pipeline and print the result
print(local_llm('translate English to German: How old are you?'))
```

## Multi-Document Retrieval with Local Language Model

The ability to extract valuable insights from multiple documents efficiently is a game-changer. Local Language Models (LLMs) have emerged as a formidable tool for this task, offering the power of advanced natural language processing right on your own machine. In this article, we’ll explore how you can use local LLMs to find information from multiple documents at once.

```python
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
```

### Loading Documents from a Directory

```python
# Create a DirectoryLoader object to load documents from a specified directory
loader = DirectoryLoader('./pdf_docs/pdf_docs/', glob="./*.pdf", loader_cls=PyPDFLoader)

# Load documents from the specified directory using the loader
documents = loader.load()

# Print the number of loaded documents
print(len(documents))
```

### Splitting Text into Chunks for Efficient Processing

```python
# Create a RecursiveCharacterTextSplitter object to split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,   # the text will be divided into chunks, and each chunk will contain up to 1000 characters.
                                               chunk_overlap=200  # the last 200 characters of one chunk will overlap with the first 200 characters of the next chunk
                                               )

# Split documents into text chunks using the text splitter
texts = text_splitter.split_documents(documents)
```

### Initializing Hugging Face Instructor Embeddings

```python
# Pass the directory path where the embedding model is stored on your system
embedding_model_name = "./hkunlp/instructor-base"

# Initialize an instance of HuggingFaceInstructEmbeddings
instructor_embeddings = HuggingFaceInstructEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={"device": "cpu"}  # Specify the device to be used for inference (GPU - "cuda" or CPU - "cpu")
)
```

### Embedding and Storing Texts with Chroma for Future Retrieval

```python
# Embed and store the texts
# Supplying a persist_directory will store the embeddings on disk

# Define the directory where the embeddings will be stored on disk
persist_directory = 'db'

# Assign the embedding model (instructor_embeddings) to the 'embedding' variable
embedding = instructor_embeddings
```

```python
# Create a Chroma instance and generate embeddings from the supplied texts
# Store the embeddings in the specified 'persist_directory' (on disk)
vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)

# Persist the database (vectordb) to disk
vectordb.persist()

# Set the vectordb variable to None to release the memory
vectordb = None
```

Now we can load the persisted database from disk and use it as normal.

```python
# Create a new Chroma instance by loading the persisted database from the 
# specified directory and using the provided embedding function.
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
```

### Configuring a Retrieval System

A retriever is an interface that returns documents given an unstructured query.

```python
# Create a retriever from the Chroma database (vectordb) with search parameters
# The value of "k" determines the number of nearest neighbors to retrieve.
retriever = vectordb.as_retriever(search_kwargs={"k": 3})
```

### Setting Up a Retrieval-Based Question-Answering System with Local Language Model and Retrieval Component

```python
# Create a Question-Answer (QA) chain for retrieval-based QA using specified components
# - 'llm' is the local language model (LLM)
# - 'chain_type' specifies the type of QA chain (e.g., "stuff")
# - 'retriever' is the retrieval component used for finding relevant documents
# - 'return_source_documents=True' indicates that source documents will be returned along with the answer.

qa_chain = RetrievalQA.from_chain_type(llm=local_llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

# Example query for the QA chain
query = "What is ReAct Prompting?"

# Use the QA chain to answer the query
llm_response = qa_chain(query)

# Print the response from the QA chain
print(llm_response)
```

## Conclusion

In this article, we have discussed how to use a local language model (LLM) for multi-document retrieval. We have presented a simple but effective approach that uses a vector store to index the documents and the LLM to retrieve the most relevant ones. We have also provided a [Colab notebook](https://colab.research.google.com/drive/1XVjur9cdYLhxe6mrQwQUlWIm4VJmiYlu?usp=sharing) with a working code to demonstrate our approach.
