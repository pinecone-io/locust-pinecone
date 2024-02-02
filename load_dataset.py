#!/usr/bin/env python3

"""
Simple program to load a dataset from 'pinecone_datasets' into a Pinecone
index.
"""

from dotenv import load_dotenv
import os
import pinecone_datasets
import sys

load_dotenv('.env')

index_name = sys.argv[1]
dataset = sys.argv[2]
apikey = os.environ['PINECONE_API_KEY']

print(f"Loading dataset '{dataset}' into Pinecone index '{index_name}'...")
ds= pinecone_datasets.load_dataset(dataset)
ds.to_pinecone_index(index_name, should_create_index=False)
