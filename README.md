<img src="pinecone-logo.png" /><img src="locust-logo.webp" height=125px/> 

# Locust load testing for Pinecone
Run load tests against your Pinecone index. This repository assumes you already have a Pinecone account, an index, and data has already been upserted. To learn more about how to write a locust file [click here](https://docs.locust.io/en/stable/writing-a-locustfile.html)

## Installation

1. clone this repo and cd to 'locust-pinecone'

2. It is highly recommended to create a virtual environment before running the requirements portion of this README.
```shell
python3 -m venv .venv
```
```shell
source .venv/bin/activate
```

3. pip install project requirements:

```shell
pip3 install -r requirements.txt
```

## Prepare the environment variables start Locust

2. run the following bash script:

```shell
bash ./locust.sh
```

This script will start with a setup shell tool which helps you configure the app.
You should provide this script the following:
1. API key Pinecone such as fb1b20bc-d702-4248-bb72-1fcd50f03616 (Your API Key is in your Pinecone console under Projects)
2. Full path to your index such as http://squad-p2-2a849c7.svc.us-east-1-aws.pinecone.io (Your API Host is found under the index section of your Pinecone console)

Note: once you configure your app, the API key and API Host will be written to a .env file in the repo,
and will be used automatically next time you run the script. You can edit this as needed if your API key or API host change in the future.

After writing the .env file, the script will then start Locust which can be accessed following the instructions in the shell
```shell
pineconeMac.local/INFO/locust.main: Starting web interface at http://0.0.0.0:8089 (accepting connections from all network interfaces)  

pineconeMac.local/INFO/locust.main: Starting Locust 2.13.1
```

The next time you run the application, it will load the environmental variables and start Locust.
```shell
bash ./locust.sh
```