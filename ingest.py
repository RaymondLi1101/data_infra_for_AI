import os
import boto3
from langchain.embeddings import LangChainEmbeddings
from langchain.text_splitter import PDFTextSplitter
import pinecone

# Configuration
BUCKET_NAME = 'your-s3-bucket-name'
PREFIX = 'your/prefix/'  # If you have a specific folder in the bucket
DOWNLOAD_PATH = '/tmp'  # Temporary download path on EC2
PINECONE_API_KEY = 'your-pinecone-api-key'
PINECONE_ENVIRONMENT = 'your-pinecone-environment'
INDEX_NAME = 'your-index-name'

# Initialize S3 client
s3 = boto3.client('s3')

def download_pdfs_from_s3():
    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=PREFIX)
    for obj in response.get('Contents', []):
        file_name = obj['Key'].split('/')[-1]
        if file_name.endswith('.pdf'):
            s3.download_file(BUCKET_NAME, obj['Key'], f"{DOWNLOAD_PATH}/{file_name}")

def process_pdfs():
    embeddings = []
    for file_name in os.listdir(DOWNLOAD_PATH):
        if file_name.endswith('.pdf'):
            file_path = f"{DOWNLOAD_PATH}/{file_name}"
            with open(file_path, 'rb') as f:
                text_splitter = PDFTextSplitter()
                texts = text_splitter.split(f)
                langchain_embeddings = LangChainEmbeddings()
                for text in texts:
                    embedding = langchain_embeddings.embed(text)
                    embeddings.append((file_name, embedding))
    return embeddings

def load_embeddings_to_pinecone(embeddings):
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    index = pinecone.Index(INDEX_NAME)
    index.upsert(vectors=embeddings)

def main():
    download_pdfs_from_s3()
    embeddings = process_pdfs()
    load_embeddings_to_pinecone(embeddings)

if __name__ == "__main__":
    main()