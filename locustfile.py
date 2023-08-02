import random
from locust import HttpUser, task
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()  # take environment variables from .env.

word_list = [x.strip() for x in open("./colornames.txt", "r")]
topK = 50
includeMetadataValue = True
includeValuesValue = False
apikey = os.environ['PINECONE_API_KEY']
dimensions = 384


class locustUser(HttpUser):
    def randomQuery(self):
        return np.random.rand(dimensions).tolist()

    # wait_time = between(1, 3)
    @task
    def vectorQuery(self):
        """
        Beware: Quality alert!!!
        While querying by randomQuery() *does* work, and will give an
        idea of performance metrics (RPS, latency, etc.), the responses
        from these queries will suffer from rather low recall as
        compared to the actual ground truth nearest neighbors,
        particularly for higher dimensionality and/or sparser vector
        space. ANNs are generally optimized for querying denser
        locations in vector space.
        """
        self.client.post("/query", name="Vector (Query only)",
                         headers={"Api-Key": apikey},
                         json={"queries": [{"values": self.randomQuery()}],
                               "topK": topK,
                               "includeMetadata": includeMetadataValue,
                               "includeValues": includeValuesValue})

    @task
    def fetchQuery(self):
        randId = random.randint(0, 85794)
        self.client.get("/vectors/fetch?ids=" + str(randId), name="Fetch",
                        headers={"Api-Key": apikey})

    @task
    def deleteById(self):
        randId = random.randint(0, 85794)
        self.client.post("/vectors/delete", name="Delete",
                         headers={"Api-Key": apikey},
                         json={"ids": [str(randId)]})

    @task
    def vectorMetadataQuery(self):
        metadata = dict(color=random.choices(word_list))
        self.client.post("/query", name="Vector + Metadata",
                         headers={"Api-Key": apikey},
                         json={"queries": [{"values": self.randomQuery()}],
                               "topK": topK,
                               "includeMetadata": includeMetadataValue,
                               "includeValues": includeValuesValue,
                               "filter": {"color": metadata['color'][0]}})

    @task
    def vectorNamespaceQuery_(self):
        self.client.post("/query", name="Vector + Namespace (namespace1)",
                         headers={"Api-Key": apikey},
                         json={"queries": [{"values": self.randomQuery()}],
                               "topK": topK,
                               "includeMetadata": includeMetadataValue,
                               "includeValues": includeValuesValue,
                               "namespace": "namespace1"})

    @task
    def vectorMetadataNamespaceQuery(self):
        metadata = dict(color=random.choices(word_list))
        self.client.get("/query", name="Vector + Metadata + Namespace (namespace1)",
                        headers={"Api-Key": apikey},
                        json={"queries": [{"values": self.randomQuery()}],
                              "topK": topK,
                              "includeMetadata": includeMetadataValue,
                              "includeValues": includeValuesValue,
                              "namespace": "namespace1",
                              "filter": {"color": metadata['color'][0]}})
