import json
import os
import pytest
import subprocess
import sys
from subprocess import PIPE


@pytest.fixture
def index_host():
    host = os.environ.get('INDEX_HOST', None)
    if host is None or host == '':
        raise Exception(
            'INDEX_HOST environment variable is not set. Set to the host of a Pinecone index suitable for testing '
            'against.')
    return host


def spawn_locust(host, mode, timeout, extra_args=[]):
    proc = subprocess.Popen(
        ["locust", "--host", host, "--iterations=1", "--headless", "--json", "--pinecone-mode",
         mode] + extra_args,
        stdout=PIPE,
        stderr=PIPE,
        text=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as e:
        # Echo whatever stdout / stderr we got so far, to aid in debugging
        if e.stdout:
            for line in e.stdout.decode(errors='replace').splitlines():
                print(line)
        if e.stderr:
            for line in e.stderr.decode(errors='replace').splitlines():
                print(line, file=sys.stderr)
        raise
    # Echo locust's stdout & stderr to our own, so pytest can capture and
    # report them on error.
    print(stdout)
    print(stderr, file=sys.stderr)
    return proc, stdout, stderr


class TestPineconeBase:
    @staticmethod
    def do_request(index_host, mode, tag, expected_name, timeout=4, extra_args=[]):
        (proc, stdout, stderr) = spawn_locust(host=index_host,
                                              mode=mode,
                                              timeout=timeout,
                                              extra_args=["--tags", tag] + extra_args)
        # Check that stderr contains the expected output, and no errors.
        assert '"Traceback' not in stderr
        # Check stats output shows expected requests occurred and they
        # succeeded.
        # With multiple processes (--processes) we see one JSON array for
        # each process, so must handle multiple JSON objects.
        stats = []
        while stdout:
            try:
                stats.append(json.loads(stdout)[0])
                break
            except json.JSONDecodeError as e:
                stdout = stdout[e.pos:]
        for s in stats:
            # Ignore empty stat sets (e.g. the master runner emits empty
            # stats)
            if s:
                assert expected_name in s['name']
                assert s['num_requests'] == 1
                assert s['num_failures'] == 0
        assert proc.returncode == 0
        return stats

class TestPinecone(TestPineconeBase):
    def test_datasets_list(self):
        # Extend timeout for listing datasets, can take longer than default 4s.
        (proc, stdout, stderr) = spawn_locust(host="unused",
                                              mode="rest",
                                              timeout=20,
                                              extra_args=["--pinecone-dataset=list"])
        # Check that stdout contains a list of datasets (don't want to hardcode
        # complete set as that just makes the test brittle if any new datasets are added).
        for dataset in ["ANN_MNIST_d784_euclidean                            60000      10000          784",
                        "langchain-python-docs-text-embedding-ada-002         3476          0         1536",
                        "quora_all-MiniLM-L6-bm25-100K                      100000      15000          384"]:
            assert dataset in stdout
        assert proc.returncode == 1

    def test_datasets_list_details(self):
        # Extend timeout for listing datasets, can take longer than default 4s.
        (proc, stdout, stderr) = spawn_locust(host="unused",
                                              mode="rest",
                                              timeout=20,
                                              extra_args=["--pinecone-dataset=list-details"])
        # Check that stdout contains at least a couple of details from the datasets (don't want to hardcode
        # complete set as that just makes the test brittle if any new datasets are added).
        for detail in ["gs://pinecone-datasets-dev/ANN_DEEP1B_d96_angular",
                        "https://github.com/erikbern/ann-benchmarks",
                        "sentence-transformers/all-MiniLM-L6-v2"]:
            assert detail in stdout
        assert proc.returncode == 1

    def test_dataset_load(self, index_host):
        # Choosing a small dataset ("only" 60,000 documents) which also
        # has a non-zero queries set.
        # We also test the --pinecone-dataset-limit option here (which has the
        # bonus effect of speeding up the test - note that complete
        # dataset loading is tested in  test_dataset_load_multiprocess).
        test_dataset = "ANN_MNIST_d784_euclidean"
        self.do_request(index_host, "sdk", 'query', 'Vector (Query only)',
                        timeout=60,
                        extra_args=["--pinecone-dataset", test_dataset,
                                    "--pinecone-dataset-limit", "123",
                                    "--pinecone-populate-index", "always"])

    def test_dataset_load_multiprocess(self, index_host):
        # Choosing a small dataset ("only" 60,000 documents) which also
        # has a non-zero queries set.
        test_dataset = "ANN_MNIST_d784_euclidean"
        self.do_request(index_host, "sdk", 'query', 'Vector (Query only)',
                        timeout=60,
                        extra_args=["--pinecone-dataset", test_dataset,
                                    "--pinecone-populate-index", "always",
                                    "--processes", "1"])


    def test_dataset_load_empty_queries(self, index_host):
        # Don't load the queries set, so we test generating queries from
        # documents.
        test_dataset = "ANN_MNIST_d784_euclidean"
        self.do_request(index_host, "sdk", 'query', 'Vector (Query only)',
                        timeout=60,
                        extra_args=["--pinecone-dataset", test_dataset,
                                    "--pinecone-populate-index", "always",
                                    "--pinecone-dataset-ignore-queries"])

    def test_recall(self, index_host):
        # Simple smoke-test for --pinecone-recall; check it is accepted
        # and no errors occur.
        test_dataset = "ANN_MNIST_d784_euclidean"
        stats = self.do_request(index_host, "sdk", 'query', 'Vector (Query only)',
                                timeout=60,
                                extra_args=["--pinecone-dataset", test_dataset,
                                            "--pinecone-recall"])

    def test_recall_requires_nearest_neighbours(self, index_host):
        # --pinecone-recall is incompatible with a dataset without
        # nearest-neighbour information in the query set - e.g. when not
        # specifying a dataset.
        (proc, _, stderr) = spawn_locust(host=index_host,
                                         mode="sdk",
                                         timeout=10,
                                         extra_args=["--tags", "query", "--pinecone-recall"])
        assert "cannot calculate Recall" in stderr
        assert proc.returncode == 1

    def test_recall_no_matches(self, index_host):
        # Test that having zero matches (e.g. namespace query where no
        # documents match) is handled correctly.
        test_dataset = "ANN_MNIST_d784_euclidean"
        stats = self.do_request(index_host, "sdk", 'query_namespace', 'Vector + Namespace',
                                timeout=60,
                                extra_args=["--pinecone-dataset", test_dataset,
                                            "--pinecone-dataset-limit", "10",
                                            "--pinecone-recall"])
        assert stats[0]["min_response_time"] == 0
        assert stats[0]["max_response_time"] == 0


@pytest.mark.parametrize("mode", ["rest", "sdk", "sdk+grpc"])
class TestPineconeModes(TestPineconeBase):
    def test_pinecone_query(self, index_host, mode):
        self.do_request(index_host, mode, 'query', 'Vector (Query only)')

    def test_pinecone_query_multiprocess(self, index_host, mode):
        self.do_request(index_host, mode, 'query', 'Vector (Query only)', timeout=20, extra_args=["--processes=1"])

    def test_pinecone_query_meta(self, index_host, mode):
        self.do_request(index_host, mode, 'query_meta', 'Vector + Metadata')

    def test_pinecone_query_namespace(self, index_host, mode):
        self.do_request(index_host, mode, 'query_namespace', 'Vector + Namespace')

    def test_pinecone_fetch(self, index_host, mode):
        self.do_request(index_host, mode, 'fetch', 'Fetch')

    def test_pinecone_delete(self, index_host, mode):
        self.do_request(index_host, mode, 'delete', 'Delete',
                        extra_args=["--pinecone-destructive-tasks"])

    def test_pinecone_limit_throughput(self, index_host, mode):
        self.do_request(index_host, mode, 'query', 'Vector (Query only)',
                        extra_args=["--pinecone-throughput-per-user=1"])

    def test_pinecone_unlimited_throughput(self, index_host, mode):
        # Test that a limit of 0 meaning "unlimited" is handled correctly.
        self.do_request(index_host, mode, 'query', 'Vector (Query only)',
                    extra_args=["--pinecone-throughput-per-user=0"])
