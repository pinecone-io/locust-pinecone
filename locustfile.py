import pathlib
import random
import time
from functools import wraps
import gevent
import grpc.experimental.gevent as grpc_gevent
from locust import FastHttpUser, User, events, tag, task
from locust.env import Environment
from locust.exception import StopUser
from locust.runners import Runner, WorkerRunner
from locust.user.task import DefaultTaskSet, TaskSet
import logging
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

colornames_file = pathlib.Path(__file__).parent / "colornames.txt"
word_list = [x.strip() for x in open(colornames_file, "r")]
includeMetadataValue = True
includeValuesValue = False
apikey = os.environ['PINECONE_API_KEY']


@events.init_command_line_parser.add_listener
def _(parser):
    parser.add_argument("--pinecone-topk", type=int, metavar="<int>", default=10,
                        help=("Number of results to return from a Pinecone "
                              "query() request. Defaults to 10."))
    parser.add_argument("--pinecone-mode", choices=["rest", "sdk", "sdk+grpc"],
                        default="rest",
                        help="How to connect to the Pinecone index (default: %(default)s). Choices: "
                             "'rest': Pinecone REST API (via a normal HTTP client). "
                             "'sdk': Pinecone Python SDK ('pinecone-client'). "
                             "'sdk+grpc': Pinecone Python SDK using gRPC as the underlying transport.")
    # iterations option included from locust-plugins
    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        help="Run at most this number of task iterations and terminate once they have finished",
        default=0,
    )


@events.test_start.add_listener
def set_up_iteration_limit(environment: Environment, **kwargs):
    options = environment.parsed_options
    if options.iterations:
        runner: Runner = environment.runner
        runner.iterations_started = 0
        runner.iteration_target_reached = False
        logging.debug(f"Iteration limit set to {options.iterations}")

        def iteration_limit_wrapper(method):
            @wraps(method)
            def wrapped(self, task):
                if runner.iterations_started == options.iterations:
                    if not runner.iteration_target_reached:
                        runner.iteration_target_reached = True
                        logging.info(
                            f"Iteration limit reached ({options.iterations}), stopping Users at the start of their next task run"
                        )
                    if runner.user_count == 1:
                        logging.info("Last user stopped, quitting runner")
                        if isinstance(runner, WorkerRunner):
                            runner._send_stats()  # send a final report
                        # need to trigger this in a separate greenlet, in case test_stop handlers do something async
                        gevent.spawn_later(0.1, runner.quit)
                    raise StopUser()
                runner.iterations_started = runner.iterations_started + 1
                method(self, task)

            return wrapped

        # monkey patch TaskSets to add support for iterations limit. Not ugly at all :)
        TaskSet.execute_task = iteration_limit_wrapper(TaskSet.execute_task)
        DefaultTaskSet.execute_task = iteration_limit_wrapper(DefaultTaskSet.execute_task)


class PineconeUser(User):
    def __init__(self, environment):
        super().__init__(environment)

        # Determine the dimensions of our index
        self.pinecone = Pinecone(apikey)
        self.index = self.pinecone.Index(host=self.host)
        self.dimensions = self.index.describe_index_stats()['dimension']

        # Set test properties from command-line args
        self.top_k = environment.parsed_options.pinecone_topk
        self.mode = environment.parsed_options.pinecone_mode
        if self.mode == "rest":
            self.client = PineconeRest(self.environment)
        elif self.mode == "sdk":
            self.client = PineconeSdk(self.environment)
        elif self.mode == "sdk+grpc":
            self.client = PineconeSdk(self.environment, use_grpc=True)
        else:
            raise Exception(f"Invalid pinecone_mode {self.mode}")

    def randomQuery(self):
        # Return random floats in the range [-1.0, 1.0], suitable
        # for using as query vector for typical embedding data.
        return ((np.random.random_sample(self.dimensions) * 2.0) - 1.0).tolist()

    @tag('query')
    @task
    def vectorQuery(self):
        self.client.query(name="Vector (Query only)",
                          q_vector=self.randomQuery(), top_k=self.top_k)

    @tag('fetch')
    @task
    def fetchQuery(self):
        randId = str(random.randint(0,85794))
        self.client.fetch(randId)

    @tag('delete')
    @task
    def deleteById(self):
        randId = str(random.randint(0,85794))
        self.client.delete(randId)

    @tag('query_meta')
    @task
    def vectorMetadataQuery(self):
        metadata = dict(color=random.choices(word_list))
        self.client.query(name="Vector + Metadata",
                          q_vector=self.randomQuery(),
                          top_k=self.top_k,
                          q_filter={"color": metadata['color'][0]})

    @tag('query_namespace')
    @task
    def vectorNamespaceQuery(self):
        self.client.query(name="Vector + Namespace (namespace1)",
                          q_vector=self.randomQuery(),
                          top_k=self.top_k,
                          namespace="namespace1")

    @tag('query_meta_namespace')
    @task
    def vectorMetadataNamespaceQuery(self):
        metadata = dict(color=random.choices(word_list))
        self.client.query(name="Vector + Metadata + Namespace (namespace1)",
                          q_vector=self.randomQuery(),
                          top_k=self.top_k,
                          q_filter={"color": metadata['color'][0]},
                          namespace="namespace1")


class PineconeRest(FastHttpUser):
    """Pinecone REST user, responsible for making requests to Pinecone via
    the base REST API.

    This is not directly instantiated / called by the locust framework, instead
    it is called via 'PineconeUser' if --pinecone-mode=rest.
    """
    abstract = True

    def __init__(self, environment):
        self.host = environment.host
        super().__init__(environment)

    def query(self, name: str, q_vector: list, top_k: int, q_filter=None, namespace=None):
        json = {"queries": [{"values": q_vector}],
                "topK": top_k,
                "includeMetadata": includeMetadataValue,
                "includeValues": includeValuesValue}
        if q_filter:
            json['filter'] = q_filter
        if namespace:
            json['namespace'] = namespace
        self.client.post("/query", name=name,
                         headers={"Api-Key": apikey},
                         json=json)

    def fetch(self, id : str):
        self.client.get("/vectors/fetch?ids=" + id, name=f"Fetch",
                        headers={"Api-Key": apikey})

    def delete(self, id : str):
        self.client.post("/vectors/delete", name=f"Delete",
                        headers={"Api-Key": apikey},
                        json={"ids": [id]})


class PineconeSdk(User):
    """Pinecone SDK client user, responsible for making requests to Pinecone via
    the 'pinecone-client' Python library.

    This is not directly instantiated / called by the locust framework, instead
    it is called via 'PineconeUser' if --pinecone-mode=rest.
    """
    abstract = True

    def __init__(self, environment, use_grpc : bool = False):
        super().__init__(environment)
        self.host = environment.host

        if use_grpc:
            self.request_type="Pine gRPC"
            self.pinecone = PineconeGRPC(apikey)
        else:
            self.request_type="Pine"
            self.pinecone = Pinecone(apikey)
        self.index = self.pinecone.Index(host=self.host)

    def query(self, name: str, q_vector: list, top_k: int, q_filter=None, namespace=None):
        args = {'vector': q_vector,
                  'top_k': top_k,
                  'include_values': includeValuesValue,
                  'include_metadata': includeValuesValue}
        if q_filter:
            args['filter'] = q_filter
        if namespace:
            args['namespace'] = namespace

        start = time.time()
        result = self.index.query(**args)
        stop = time.time()

        response_time = (stop - start) * 1000.0
        match_count = len(result.matches)

        events.request.fire(request_type=self.request_type,
                            name=name,
                            response_length=match_count,
                            response_time=response_time)

    def fetch(self, id : str):
        start = time.time()
        self.index.fetch(ids=[id])
        stop = time.time()

        response_time = (stop - start) * 1000.0
        events.request.fire(request_type=self.request_type,
                            name="Fetch",
                            response_length=0,
                            response_time=response_time)

    def delete(self, id : str):
        start = time.time()
        self.index.delete(ids=[id])
        stop = time.time()

        response_time = (stop - start) * 1000.0
        events.request.fire(request_type=self.request_type,
                            name="Delete",
                            response_length=0,
                            response_time=response_time)
