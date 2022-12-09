#!/bin/bash
set -e

# set up index and .env file
python3 interactive_setup.py

# run locust app
source .env
locust -H $PINECONE_API_HOST
