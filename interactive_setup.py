import os
import typer
import pandas as pd
from typing import List
from pathlib import Path
from dotenv import load_dotenv
from rich import print
from rich.prompt import Prompt

load_dotenv()  # take environment variables from .env.


def main():

    print("Starts app demo setup. This script is going to write to your .env file to save time on your next app run\n")

    evn_lines_to_append: List[str] = []

    if "PINECONE_API_KEY" in os.environ:
        print(f"[green]Found Pinecone API key in .env file, prefix: {os.environ['PINECONE_API_KEY']}\n[/green]")
        api_key = os.environ["PINECONE_API_KEY"]
    else:
        api_key = Prompt.ask("Please enter Pinecone API key")
        evn_lines_to_append.append(f"PINECONE_API_KEY={api_key}")
    
    if "PINECONE_API_HOST" in os.environ:
        print(f"[green]Found Pinecone API host in .env file, prefix: {os.environ['PINECONE_API_HOST']}\n[/green]")
        api_key = os.environ["PINECONE_API_HOST"]
    else:
        api_host = Prompt.ask("Please enter Pinecone API host")
        evn_lines_to_append.append(f"PINECONE_API_HOST={api_host}")

    if len(evn_lines_to_append) > 0:
        print("[green]writing to .env file\n[/green]")
        with open(".env", "a") as f:
            lines = [l + "\n" for l in [''] + evn_lines_to_append]
            f.writelines(lines)

    print("[green]Setup is done. You can edit your API key and API host in the .env file in the root directory.\n[green]")
    print("[green]Starting Locust...\n[green]")


if __name__ == '__main__':
    typer.run(main)
