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


def spawn_locust(host, mode, extra_args):
    proc = subprocess.Popen(
        ["locust", "--host", host, "--iterations=1", "--headless", "--json", "--pinecone-mode",
         mode] + extra_args,
        stdout=PIPE,
        stderr=PIPE,
        text=True,
    )
    stdout, stderr = proc.communicate(timeout=4)
    # Echo locust's stdout & stderr to our own, so pytest can capture and
    # report them on error.
    print(stdout)
    print(stderr, file=sys.stderr)
    return proc, stdout, stderr


@pytest.mark.parametrize("mode", ["rest", "sdk", "sdk+grpc"])
class TestPinecone:
    @staticmethod
    def do_request(index_host, mode, tag, expected_name):
        (proc, stdout, stderr) = spawn_locust(index_host, mode, ["--tags", tag])
        # Check that stderr contains the expected output, and no errors.
        assert '"Traceback' not in stderr
        assert 'All users spawned: {"PineconeUser": 1}' in stderr
        # Check stats output shows expected requests occurred and they
        # succeeded.
        stats = json.loads(stdout)[0]
        assert expected_name in stats['name']
        assert stats['num_requests'] == 1
        assert stats['num_failures'] == 0
        assert proc.returncode == 0

    def test_pinecone_query(self, index_host, mode):
        self.do_request(index_host, mode, 'query', 'Vector (Query only)')

    def test_pinecone_query_meta(self, index_host, mode):
        self.do_request(index_host, mode, 'query_meta', 'Vector + Metadata')

    def test_pinecone_query_namespace(self, index_host, mode):
        self.do_request(index_host, mode, 'query_namespace', 'Vector + Namespace')

    def test_pinecone_fetch(self, index_host, mode):
        self.do_request(index_host, mode, 'fetch', 'Fetch')

    def test_pinecone_delete(self, index_host, mode):
        self.do_request(index_host, mode, 'delete', 'Delete')
