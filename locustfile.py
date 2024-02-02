import random
import time
import grpc.experimental.gevent as grpc_gevent
from locust import FastHttpUser, User, events, tag, task
import numpy as np
from dotenv import load_dotenv
import os
from pinecone import Pinecone
from pinecone.grpc import PineconeGRPC


# patch grpc so that it uses gevent instead of asyncio. This is required to
# allow the multiple coroutines used by locust to run concurrently. Without it
# (using default asyncio) will block the whole Locust/Python process,
# in practice limiting to running a single User per worker process.
grpc_gevent.init_gevent()

load_dotenv()  # take environment variables from .env.

word_list = [x.strip() for x in open("./colornames.txt", "r")]
includeMetadataValue = True
includeValuesValue = False
apikey = os.environ['PINECONE_API_KEY']

@events.init_command_line_parser.add_listener
def _(parser):
    parser.add_argument("--pinecone-topk", type=int, default=10,
                        help=("Number of results to return from a Pinecone "
                              "query() request. Defaults to 10."))


class PineconeRest(FastHttpUser):
    def __init__(self, environment):
        super().__init__(environment)

        # Determine the dimensions of our index
        self.pinecone = Pinecone(apikey)
        self.index = self.pinecone.Index(host=self.host)
        self.dimensions =self.index.describe_index_stats()['dimension']

        # Set test properties from command-line args
        self.top_k = environment.parsed_options.pinecone_topk

    def randomQuery(self):
        # Return random floats in the range [-1.0, 1.0], suitable
        # for using as query vector for typical embedding data.
        return ((np.random.random_sample(self.dimensions) * 2.0) - 1.0).tolist()

    #wait_time = between(1, 3)
    @tag('query')
    @task
    def vectorQuery(self):
        self.client.post("/query", name=f"Vector (Query only)",
                        headers={"Api-Key": apikey},
                        json={"queries": [{"values": self.randomQuery()}],
                              "topK": self.top_k,
                              "includeMetadata": includeMetadataValue,
                              "includeValues": includeValuesValue})

    @tag('fetch')
    @task
    def fetchQuery(self):
        randId = random.randint(0,85794)
        self.client.get("/vectors/fetch?ids=" + str(randId), name=f"Fetch",
                        headers={"Api-Key": apikey})

    @tag('delete')
    @task
    def deleteById(self):
        randId = random.randint(0,85794)
        self.client.post("/vectors/delete", name=f"Delete",
                        headers={"Api-Key": apikey},
                        json={"ids": [str(randId)]})

    @tag('query_meta')
    @task
    def vectorMetadataQuery(self):
        metadata = dict(color=random.choices(word_list))
        self.client.post("/query", name=f"Vector + Metadata",
                        headers={"Api-Key": apikey},
                        json={"queries": [{"values": self.randomQuery()}],
                              "topK": self.top_k,
                              "includeMetadata": includeMetadataValue,
                              "includeValues": includeValuesValue,
                              "filter": {"color": metadata['color'][0]}})

    @tag('query_namespace')
    @task
    def vectorNamespaceQuery_(self):
        self.client.post("/query", name=f"Vector + Namespace (namespace1)",
                        headers={"Api-Key": apikey},
                        json={"queries": [{"values": self.randomQuery()}],
                              "topK": self.top_k,
                              "includeMetadata": includeMetadataValue,
                              "includeValues": includeValuesValue,
                              "namespace": "namespace1"})

    @tag('query_meta_namespace')
    @task
    def vectorMetadataNamespaceQuery(self):
        metadata = dict(color=random.choices(word_list))
        self.client.get("/query", name=f"Vector + Metadata + Namespace (namespace1)",
                        headers={"Api-Key": apikey},
                        json={"queries": [{"values": self.randomQuery()}],
                              "topK": self.top_k,
                              "includeMetadata": includeMetadataValue,
                              "includeValues": includeValuesValue,
                              "namespace": "namespace1",
                              "filter": {"color": metadata['color'][0]}})


class PineconeGrpc(User):
    def __init__(self, environment):
        super().__init__(environment)

        # Determine the dimensions of our index
        self.pinecone = PineconeGRPC(apikey)
        self.index = self.pinecone.Index(host=self.host)
        self.dimensions =self.index.describe_index_stats()['dimension']

        # Set test properties from command-line args
        self.top_k = environment.parsed_options.pinecone_topk

    def randomQuery(self):
        # Return random floats in the range [-1.0, 1.0], suitable
        # for using as query vector for typical embedding data.
        return ((np.random.random_sample(self.dimensions) * 2.0) - 1.0).tolist()

    @tag('query')
    @task
    def vectorQuery(self):
        args = {'vector': self.randomQuery(),
                  'top_k': self.top_k,
                  'include_values': includeValuesValue,
                  'include_metadata': includeValuesValue}
        start = time.time()
        result = self.index.query(**args)
        stop = time.time()

        response_time = (stop - start) * 1000.0
        match_count = len(result.matches)

        events.request.fire(request_type="gRPC",
                            name="query",
                            response_length=match_count,
                            response_time=response_time)
