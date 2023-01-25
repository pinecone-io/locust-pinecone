import random
from locust import HttpUser, task
import numpy as np
from dotenv import load_dotenv
import os
import json

load_dotenv()  # take environment variables from .env.

word_list = [x.strip() for x in open("./colornames.txt", "r")]
queries = [x.strip() for x in open("./sample-sbert-vectors.txt", "r")]
topK = 100
includeMetadataValue = True
includeValuesValue = False
apikey = os.environ['PINECONE_API_KEY']
dimensions = 768

class locustUser(HttpUser):
    def randomQuery(self):
        #return np.random.rand(dimensions).tolist()
        return json.loads(random.choices(queries)[0])

    #wait_time = between(1, 3)
    @task
    def vectorQuery(self):
        self.client.post("/query", name=f"Vector (Query only)",
                        headers={"Api-Key": apikey},
                        json={"queries": [{"values": self.randomQuery()}],
                              "topK": topK,
                              "includeMetadata": includeMetadataValue,
                              "includeValues": includeValuesValue})

    # @task(1)
    # def fetchQuery(self):
    #     randId = random.randint(0,85794)
    #     self.client.get("/vectors/fetch?ids=" + str(randId), name=f"Fetch",
    #                     headers={"Api-Key": apikey})

    # @task(1)
    # def deleteById(self):
    #     randId = random.randint(0,85794)
    #     self.client.post("/vectors/delete", name=f"Delete",
    #                     headers={"Api-Key": apikey},
    #                     json={"ids": [str(randId)]})

    @task
    def vectorMetadataQuery(self):
        metadata = dict(color=random.choices(word_list))
        self.client.post("/query", name=f"Vector (Query + Metadata Filter)",
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