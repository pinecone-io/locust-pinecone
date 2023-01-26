import random
from locust import HttpUser, task
import numpy as np
from dotenv import load_dotenv
import os
import json

load_dotenv()  # take environment variables from .env.

#word_list = [x.strip() for x in open("./colornames.txt", "r")]
queries = [x.strip() for x in open("./sample-sbert-vectors.txt", "r")]
topK = 100
includeMetadataValue = True
includeValuesValue = False
apikey = os.environ['PINECONE_API_KEY']
dimensions = 768
response = json.load(open("./response.json"))
records = [match for id in response['results'] for match in id['matches']]

class locustUser(HttpUser):
    def randomQuery(self):
        #return np.random.uniform(-1, 1, dimensions).tolist()
        return json.loads(random.choices(queries)[0])

    #wait_time = between(1, 3)
    @task
    def vectorQuery(self):
        self.client.post("/query", name=f"Vector (Random Query)",
                        headers={"Api-Key": apikey},
                        json={"queries": [{"values": np.random.uniform(-1, 1, dimensions).tolist()}],
                              "topK": topK,
                              "includeMetadata": includeMetadataValue,
                              "includeValues": includeValuesValue})

    @task
    def fetchQuery(self):
        self.client.get("/vectors/fetch?ids=" + random.choices(records,k=1)[0]['id'], name=f"Fetch by known ID",
                        headers={"Api-Key": apikey})

    @task
    def deleteById(self):
        self.client.post("/vectors/delete", name=f"Delete",
                        headers={"Api-Key": apikey},
                        json={"ids": [random.choices(records,k=1)[0]['id']]})

    @task
    def vectorUpsert(self):
        record = [random.choices(records,k=1)[0]]
        self.client.post("/vectors/upsert", name=f"Upsert",
                        headers={"Api-Key": apikey},
                        json={"vectors": [{"values": record[0]['values'],"metadata": record[0]['metadata'],"id": record[0]['id']}]})

    @task
    def vectorMetadataQuery(self):
        #metadata = dict(color=random.choices(word_list))
        self.client.post("/query", name=f"Vector (Common Query + Metadata Filter)",
                        headers={"Api-Key": apikey},
                        json={"queries": [{"values": self.randomQuery()}],
                              "topK": topK,
                              "includeMetadata": includeMetadataValue,
                              "includeValues": includeValuesValue,
                              "filter": {'$or': [{'$and': [{'question_type': {'$eq': 'document_questions'}}, {'answered': {'$eq': True}}]}, {'$and': [{'question_type': {'$eq': 'user_questions'}}, {'answered': {'$eq': True}}, {'has_question_attachments': {'$eq': False}}]}]}})

    ## TODO Create a case for upserts


    # @task
    # def vectorNamespaceQuery_(self):
    #     self.client.post("/query", name=f"Vector + Namespace (namespace1)",
    #                     headers={"Api-Key": apikey},
    #                     json={"queries": [{"values": self.randomQuery()}],
    #                           "topK": topK,
    #                           "includeMetadata": includeMetadataValue,
    #                           "includeValues": includeValuesValue,
    #                           "namespace": "namespace1"})

    # @task
    # def vectorMetadataNamespaceQuery(self):
    #     metadata = dict(color=random.choices(word_list))
    #     self.client.get("/query", name=f"Vector + Metadata + Namespace (namespace1)",
    #                     headers={"Api-Key": apikey},
    #                     json={"queries": [{"values": self.randomQuery()}],
    #                           "topK": topK,
    #                           "includeMetadata": includeMetadataValue,
    #                           "includeValues": includeValuesValue,
    #                           "namespace": "namespace1",
    #                           "filter": {"color": metadata['color'][0]}})