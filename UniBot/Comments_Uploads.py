# Import necessary libraries
import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from transformers import pipeline
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer

# Function to fetch text content from a website URL
def get_website_text(url):
    response = requests.get(url)
    html_content = response.text
    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text(separator="\n", strip=True)
    return text

# Function to extract all links from a website
def get_all_links(site):
    url = site
    reqs = requests.get(url)
    soup = BeautifulSoup(reqs.text, 'html.parser')
    
    urls = []
    for link in soup.find_all('a'):
        site = link.get('href')
        if site is not None:
            if site.startswith("https"):
                urls.append(site)
    return urls

# Function to fetch text data from all URLs provided
def get_all_data_with_links(urls):
    arr_dict = []
    for url in urls:
        data_dict = {}
        data_dict["reference"] = url
        data_dict["html_text"] = get_website_text(url)
        arr_dict.append(data_dict)
        
    return arr_dict

# Class for chunking text data
class ChunkData:
    def __init__(self, chunk_size=300, chunk_overlap=50):
        chunk_size = chunk_size
        chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
                               
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap,
                                length_function=len,
                        )
    def create_chunks(self, text):
        chunks = self.text_splitter.create_documents(
                texts=[text["html_text"]], 
                metadatas=[{"source": text["reference"]
                   }])
        return [{"text": chunk.page_content.replace('\n', ""), "source": chunk.metadata["source"]} for chunk in chunks]

# Function to scrape a website and chunk its content
def scrap_and_chunk(url):
    urls = get_all_links("https://lpu.in")
    print(len(urls))
    all_data = get_all_data_with_links(urls)
    chunkData = ChunkData(chunk_size=500)
    chunks = chunkData.create_chunks(all_data[0])
    chunk_list = chunkData.create_chunks(all_data[0])
    for i in range(1,len(all_data)):
        c_data = chunkData.create_chunks(all_data[i])
        for c in c_data:
            chunk_list.append(c)
    return chunk_list

# Function to insert chunked data into Qdrant
def insert_into_qdrant():
    chunk_list = scrap_and_chunk("https://lpu.in")
    qdrant.recreate_collection(
    collection_name="university_data",
    vectors_config=models.VectorParams(
        size=encoder.get_sentence_embedding_dimension(),  # Vector size is defined by used model
        distance=models.Distance.COSINE,
        ),
    )

    qdrant.upload_records(
    collection_name="university_data",
    records=[
        models.Record(
            id=idx, vector=encoder.encode(doc["text"]).tolist(), payload=doc
        )
        for idx, doc in enumerate(chunk_list)
    ],
    )

# Entry point of the script
if __name__ == "__main__":
    # Initialize Qdrant client and Sentence Transformer model
    qdrant = QdrantClient("http://localhost:6333")
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Print message
    print("insert data into Qdrant")
    # Call function to insert data into Qdrant
    insert_into_qdrant()
