from dataset import Dataset

import argparse
import pathlib
import random
import time
from functools import wraps
import gevent
import grpc.experimental.gevent as grpc_gevent
from locust import FastHttpUser, User, constant_throughput, events, tag, task
from locust.env import Environment
from locust.exception import StopUser
from locust.runners import Runner, WorkerRunner
import locust.stats
from locust.user.task import DefaultTaskSet, TaskSet
import logging
import numpy as np
from dotenv import load_dotenv
import os
from pinecone import Pinecone
from pinecone.grpc import PineconeGRPC
import psutil
import tempfile
import tabulate
import sys

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

# Configure pyarrow's embedded jemalloc library to immediately return
# deallocted memory back to the OS.
# (We only use pyarrow to load the dataset's Parquet files into
# pandas.DateFrames, which requires ~1GB RAM per 100k ~1000 dimension
# vectors; with the default setting pyarrow can hold this memory
# resident for extended periods after we have finished with the
# associated DataFrame, resulting in excessive process RSS,
# particulary with high process counts).
import pyarrow
try:
    pyarrow.jemalloc_set_decay_ms(0)
except NotImplementedError:
    # Raised if jemalloc is not supported by the pyarrow installation - skip
    pass


@events.init_command_line_parser.add_listener
def _(parser):
    pc_options = parser.add_argument_group("Pinecone-specific options")
    pc_options.add_argument("--pinecone-topk", type=int, metavar="<int>", default=10,
                            help=("Number of results to return from a Pinecone "
                                  "query() request. Defaults to 10."))
    pc_options.add_argument("--pinecone-mode", choices=["rest", "sdk", "sdk+grpc"],
                            default="sdk+grpc",
                            help="How to connect to the Pinecone index (default: %(default)s). Choices: "
                                 "'rest': Pinecone REST API (via a normal HTTP client). "
                                 "'sdk': Pinecone Python SDK ('pinecone-client'). "
                                 "'sdk+grpc': Pinecone Python SDK using gRPC as the underlying transport.")
    pc_options.add_argument("--pinecone-dataset", type=str, metavar="<dataset_name> | 'list' | 'list-details'", default=None,
                            help="The dataset to use for index population and/or query generation. "
                                 "Pass the value 'list' to list available datasets, or pass 'list-details' to"
                                 " list full details of available datasets.")
    pc_options.add_argument("--pinecone-dataset-ignore-queries", action=argparse.BooleanOptionalAction,
                            help="Ignore and do not load the 'queries' table from the specified dataset.")
    pc_options.add_argument("--pinecone-dataset-limit", type=int, default=0,
                            help="If non-zero, limit the dataset to the first N vectors.")
    pc_options.add_argument("--pinecone-dataset-docs-sample-for-query", type=float, default=0.01,
                            metavar="<fraction> (0.0 - 1.0)",
                            help="Specify the fraction of docs which should be sampled when the documents vectorset "
                                 "is used for queries (default: %(default)s).")
    pc_options.add_argument("--pinecone-populate-index", choices=["always", "never", "if-count-mismatch"],
                            default="if-count-mismatch",
                            help="Should the index be populated with the dataset before issuing requests. Choices: "
                                 "'always': Always populate from dataset. "
                                 "'never': Never populate from dataset. "
                                 "'if-count-mismatch': Populate if the number of items in the index differs from the "
                                 "number of items in th dataset, otherwise skip population. "
                                 "(default: %(default)s).")
    pc_options.add_argument("--pinecone-recall", action=argparse.BooleanOptionalAction,
                            help="Report the Recall score (out of 100) instead of latency (reported on UI / console as 'latency'")
    pc_options.add_argument("--pinecone-dataset-cache", type=str, default=".dataset_cache",
                            help="Path to directory to cache downloaded datasets (default: %(default)s).")
    pc_options.add_argument("--pinecone-throughput-per-user", type=float, default=0,
                            help="How many requests per second each user should issue (default: %(default)s). "
                                 "Setting to zero will make each user issue requests as fast as possible "
                                 "(next request sent as soon as previous one completes).")

    # iterations option included from locust-plugins
    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        help="Run at most this number of task iterations and terminate once they have finished",
        default=0,
    )


@events.init.add_listener
def on_locust_init(environment, **_kwargs):
    if not isinstance(environment.runner, WorkerRunner):
        # For Worker runners dataset setup is deferred until the test starts,
        # to avoid multiple processes trying to downlaod at the same time.
        setup_dataset(environment)


def setup_dataset(environment: Environment, skip_download_and_populate: bool = False):
    """
    Sets up the dataset specified via the --pinecone_dataset argument:
     - downloads it if not already present in local cache
     - reads 'documents' data (if needed for population)
     - reads 'queries' data if present.
     - populates the index with the documents data (if requested).
    The Dataset is assigned to `environment.dataset` for later use by Users
    making requests.
    """
    dataset_name = environment.parsed_options.pinecone_dataset
    if not dataset_name:
        environment.dataset = Dataset()
        return
    if dataset_name in ("list", "list-details"):
        # Print out the list of available datasets, then exit.
        print("Fetching list of available datasets for --pinecone-dataset...")
        available = Dataset.list()
        if dataset_name == "list":
            # Discard the more detailed fields before printing, however
            # copy the 'dimensions' model field from 'dense_model' into the top
            # level as that's a key piece of information (needed to create an
            # index).
            brief = []
            for a in available:
                a['dimension'] = a['dense_model']['dimension']
                summary = {key.capitalize(): a[key] for key in ['name', 'documents', 'queries', 'dimension']}
                brief.append(summary)
            available = brief
        print(tabulate.tabulate(available, headers="keys", tablefmt="simple"))
        print()
        sys.exit(1)

    logging.info(f"Loading Dataset {dataset_name} into memory for Worker {os.getpid()}...")
    environment.dataset = Dataset(dataset_name, environment.parsed_options.pinecone_dataset_cache)
    ignore_queries = environment.parsed_options.pinecone_dataset_ignore_queries
    sample_ratio = environment.parsed_options.pinecone_dataset_docs_sample_for_query
    limit = environment.parsed_options.pinecone_dataset_limit
    environment.dataset.load(skip_download=skip_download_and_populate,
                             load_queries=not ignore_queries,
                             limit=limit,
                             doc_sample_fraction=sample_ratio)
    populate = environment.parsed_options.pinecone_populate_index
    if not skip_download_and_populate and populate != "never":
        logging.info(
            f"Populating index {environment.host} with {len(environment.dataset.documents)} vectors from dataset '{dataset_name}'")
        environment.dataset.upsert_into_index(environment.host, apikey,
                                              skip_if_count_identical=(populate == "if-count-mismatch"))
    # We no longer need documents - if we were populating then that is
    # finished, and if not populating then we don't need documents for
    # anything else.
    environment.dataset.prune_documents()


@events.test_start.add_listener
def setup_worker_dataset(environment, **_kwargs):
    # happens only once in headless runs, but can happen multiple times in web ui-runs
    # in a distributed run, the master does not typically need any test data
    if isinstance(environment.runner, WorkerRunner):
        # Make the Dataset available for WorkerRunners (non-Worker will have
        # already setup the dataset via on_locust_init).
        #
        # We need to perform this work in a background thread (not in
        # the current gevent greenlet) as otherwise we block the
        # current greenlet (pandas data loading is not
        # gevent-friendly) and locust's master / worker heartbeating
        # thinks the worker has gone missing and can terminate it.
        pool = gevent.get_hub().threadpool
        environment.setup_dataset_greenlet = pool.apply_async(setup_dataset,
                                                              kwds={'environment':environment,
                                                                    'skip_download_and_populate':True})


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


@events.test_stop.add_listener
def on_stop_mem_usage(**kwargs):
    print_mem_usage("On test stop")


def print_mem_usage(label=""):
    if label:
        label = " - " + label
    process = psutil.Process()
    info = process.memory_info()

    def mb(b):
        return f"{int(b / 1024 / 1024)}MB"

    arrow_alloc = pyarrow.total_allocated_bytes()
    logging.debug(f"Memory usage for pid:{process.pid} RSS:{mb(info.rss)} "
                  f"VSZ:{mb(info.vms)} "
                  f"pyarrow.allocated:{mb(arrow_alloc)}{label}")


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
        self.target_throughput = environment.parsed_options.pinecone_throughput_per_user

        if isinstance(self.environment.runner, WorkerRunner):
            # Wait until the datset has been loaded for this environment (Runner)
            environment.setup_dataset_greenlet.join()

        # Check for compatibility between different options.
        # --pinecone-recall can only be used if the query set contains the
        # exact top-K vectors.
        if environment.parsed_options.pinecone_recall:
            query = self._query_vector()
            if "blob" not in query or "nearest_neighbors" not in query["blob"]:
                logging.error(
                    "--pinecone-recall specified but query set does not "
                    "contain nearest neighbours - cannot calculate Recall")
                sys.exit(1)

    def wait_time(self):
        if self.target_throughput > 0:
            return constant_throughput(self.target_throughput)(self)
        return 0

    @tag('query')
    @task
    def vectorQuery(self):
        self.client.query(name="Vector (Query only)",
                          query=self._query_vector(), top_k=self.top_k)

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
                          query=self._query_vector(),
                          top_k=self.top_k,
                          q_filter={"color": metadata['color'][0]})

    @tag('query_namespace')
    @task
    def vectorNamespaceQuery(self):
        self.client.query(name="Vector + Namespace (namespace1)",
                          query=self._query_vector(),
                          top_k=self.top_k,
                          namespace="namespace1")

    @tag('query_meta_namespace')
    @task
    def vectorMetadataNamespaceQuery(self):
        metadata = dict(color=random.choices(word_list))
        self.client.query(name="Vector + Metadata + Namespace (namespace1)",
                          query=self._query_vector(),
                          top_k=self.top_k,
                          q_filter={"color": metadata['color'][0]},
                          namespace="namespace1")

    def _query_vector(self):
        """
        Sample a random entry from the queries set (if non-empty), otherwise
        generate a random query vector in the range [-1.0, 1.0].
        """
        if not self.environment.dataset.queries.empty:
            record = self.environment.dataset.queries.sample(n=1).iloc[0]
        else:
            record = dict()
            record["vector"] = ((np.random.random_sample(self.dimensions) * 2.0) - 1.0).tolist()
        return record


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

    def query(self, name: str, query: dict, top_k: int, q_filter=None, namespace=None):
        json = {"vector": query["vector"],
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
            self.request_type="Pinecone gRPC"
            self.pinecone = PineconeGRPC(apikey)
        else:
            self.request_type="Pinecone"
            self.pinecone = Pinecone(apikey)
        # Ensure stats 'Type' column is wide enough for our chosen mode so
        # tables render correctly aligned (by default is only 8 chars wide).
        locust.stats.STATS_TYPE_WIDTH = len(self.request_type) + 1

        self.index = self.pinecone.Index(host=self.host)

    def query(self, name: str, query: dict, top_k: int, q_filter=None, namespace=None):
        args = {'vector': query['vector'],
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

        if self.environment.parsed_options.pinecone_recall:
            expected_neighbours = query['blob']['nearest_neighbors']
            actual_neighbours = [r['id'] for r in result.matches]
            recall_n = Dataset.recall(actual_neighbours, expected_neighbours)
            metric = recall_n * 100
        else:
            metric = response_time

        events.request.fire(request_type=self.request_type,
                            name=name,
                            response_length=match_count,
                            response_time=metric)

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
