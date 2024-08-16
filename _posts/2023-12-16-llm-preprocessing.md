---
title: "Is Preprocessing Essential for Maximizing the Potential of Large Language Models (LLMs)?"
date: 2023-12-16
tags: [LLM, Preprocessing, Processing, Generative AI, Langchain, OpenAI]
excerpt: "Is Preprocessing Essential for Maximizing the Potential of Large Language Models (LLMs)?"
header:
  overlay_image: "/images/local-llm-preprocess/home-page.jpg"
  caption: "Photo by Markus Spiske on Unsplash"
mathjax: "true"
---

## Introduction
This article delves into a real-world use case encountered during a client project, aiming to streamline manual efforts through the implementation of Gen AI, specifically OpenAI’s LLM (GPT-3.5).

## The Initial Impression
The prevailing notion was that feeding any type of data (text) into ChatGPT would yield a response, albeit not always 100% accurate. This confidence prompted the integration of OpenAI’s LLM into the application.

## The Challenge
The application’s primary task was processing invoice data — extracting crucial parameters such as items, discounts, and totals and structuring them for further processing. Invoices from various vendors in multiple formats were processed using LLM.

## Optimizing Prompts
To extract accurate data, the team had to engineer prompts that were optimized for various scenarios. Handling data returned in tabular or list format was essential, and the team standardized the output into a common structure — JSON. Below is an example of the prompt designed in the application:

```python
items_query = f"""Given the invoice data, extract details for items.
                Format the output in JSON as a list of dictionaries where the keys are:
                - item_description (also denoted as the product name),
                - quantity (assign '1' if data is not present, also denoted by 'qty'),
                - total_price (assign '$0' if data is not present)
                ..."""
```

## The Unexpected Hurdle
Despite successful results in individual cases, a significant challenge arose when processing a file containing multiple invoices. While the team expected LLM to extract parameters for all 30 invoices, the reality was quite different — it only extracted data for 5 pages. Below is the code snippet:

```python
# Import necessary modules
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings()

# Initialize ChatOpenAI model with specific parameters
llm = ChatOpenAI(temperature=0.0, model_name="gpt-4")

# Load the question-answering chain with the given model and chain type
chain = load_qa_chain(llm, chain_type="stuff")

# Initialize a CharacterTextSplitter with specific parameters
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,
)

# Define raw text containing invoice information
raw_text = """invoice text\nextracted in strings\ndata of 30 invoices"""

# Split the raw text into chunks using the defined text splitter
docs = text_splitter.split_text(raw_text)

# Create a FAISS vector store from the text chunks using the specified embeddings
db = FAISS.from_texts(docs, embeddings)

# Define a query for the similarity search
query = "how many invoices are present in the document? Generate output in json format with keys no_of_invoices"

# Perform a similarity search on the vector store with the query
docs = db.similarity_search(query)

# Run the question-answering chain on the input documents and query
list_desc = chain.run(input_documents=docs, question=query)

# Print the resulting list of descriptions
print(list_desc)
```

```
{
"no_of_invoices": 5
}
```

## Investigation and Experimentation
The team delved into an analysis, discovering that 25 out of 30 pages returned blank results, even though the data was meaningful and not noisy. Surprisingly, testing each invoice separately with ChatGPT and LLM yielded positive results.

## Token Usage and Model Switching
Initially suspecting token limitations, the team ruled out this possibility, confirming the use of a 16k model. Experimenting with different LLM models provided no resolution to the issue.

## The Game-Changer: Preprocessing
The breakthrough came when the team explored preprocessing as a solution. Through experiments with prompts and models yielded no improvement, preprocessing the data before feeding it into LLM proved to be the missing link. It allowed the application to deliver results for all 30 invoices.

While the specifics of the complete preprocessing steps applied remain confidential due to client data sensitivity, a basic example showcases the impact of this approach.

## Addressing “\n” in Invoice Text
One particular obstacle the team encountered was the presence of line breaks (“\n”) in the invoice text. This seemingly minor issue turned out to be a major stumbling block in achieving consistent results. Through careful handling of these line breaks in the preprocessing stage, the application’s performance witnessed a remarkable transformation.

```python
# Import necessary modules
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings()

# Initialize ChatOpenAI model with specific parameters
llm = ChatOpenAI(temperature=0.0, model_name="gpt-4")

# Load the question-answering chain with the given model and chain type
chain = load_qa_chain(llm, chain_type="stuff")

# Initialize a CharacterTextSplitter with specific parameters
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,
)

# Define raw text containing invoice information
raw_text = """invoice text\nextracted in strings\ndata of 30 invoices"""

# Split the raw text into chunks using the defined text splitter,
# replacing "\n" with "\\n" for proper handling
docs = text_splitter.split_text(raw_text.replace("\n","\\n"))

# Create a FAISS vector store from the text chunks using the specified embeddings
db = FAISS.from_texts(docs, embeddings)

# Define a query for the similarity search
query = "how many invoices are present in the document? Generate output in json format with keys no_of_invoices"

# Perform a similarity search on the vector store with the query
docs = db.similarity_search(query)

# Run the question-answering chain on the input documents and query
list_desc = chain.run(input_documents=docs, question=query)

# Print the resulting list of descriptions
print(list_desc)
```

```python
{
"no_of_invoices": 30
}
```

Following this small yet crucial preprocessing adjustment, the application successfully produced results for all 30 invoices. Although it’s worth noting that a few invoices still presented inaccuracies, this setback became an actionable item for further refinement. The breakthrough achieved through preprocessing not only resolved the initial challenge but also opened avenues for addressing and enhancing other aspects of the data processing pipeline.

Conclusion
In summary, while the specific preprocessing steps remain confidential, this case study highlights the crucial role of preprocessing in improving the performance of Large Language Models (LLMs). It emphasizes the need for careful prompt tuning, model selection, and the recognition of preprocessing as a vital step to fully harness the capabilities of LLMs in real-world applications. The key lies in tailoring preprocessing strategies to the unique characteristics of the data, showcasing the significant impact of a nuanced approach.

While this solution may not be universally applicable to all applications, it is certainly worth considering and experimenting with in your specific context.

## References
https://platform.openai.com/docs/models

https://platform.openai.com/docs/guides/prompt-engineering