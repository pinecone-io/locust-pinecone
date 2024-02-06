import os
import pytest
import subprocess
from subprocess import PIPE


@pytest.fixture
def index_host():
    host = os.environ.get('INDEX_HOST', None)
    if host is None or host == '':
        raise Exception('INDEX_HOST environment variable is not set')
    return host


def spawn_locust(host, mode, extra_args):
    return subprocess.Popen(
        ["locust", "--host", host, "--iterations=1", "--headless", "--pinecone-mode",
         mode] + extra_args,
        stdout=PIPE,
        stderr=PIPE,
        text=True,
    )


@pytest.mark.parametrize("mode", ["rest", "sdk", "sdk+grpc"])
class TestPinecone:

    def test_pinecone_query(self, index_host, mode):
        proc = spawn_locust(index_host, mode, ["--tags=query"])
        stdout, stderr = proc.communicate(timeout=4)
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
        assert "Traceback" not in stderr
        assert 'All users spawned: {"PineconeUser": 1}' in stderr
        assert 'Vector (Query only)' in stderr
        assert proc.returncode == 0

    def test_pinecone_query_meta(self, index_host, mode):
        proc = spawn_locust(index_host, mode, ["--tags=query_meta"])
        stdout, stderr = proc.communicate(timeout=4)
        assert "Traceback" not in stderr
        assert 'All users spawned: {"PineconeUser": 1}' in stderr
        assert 'Vector + Metadata' in stderr
        assert proc.returncode == 0

    def test_pinecone_query_namespace(self, index_host, mode):
        proc = spawn_locust(index_host, mode, ["--tags=query_namespace"])
        stdout, stderr = proc.communicate(timeout=4)
        assert "Traceback" not in stderr
        assert 'All users spawned: {"PineconeUser": 1}' in stderr
        assert 'Vector + Namespace' in stderr
        assert proc.returncode == 0

    def test_pinecone_fetch(self, index_host, mode):
        proc = spawn_locust(index_host, mode, ["--tags=fetch"])
        stdout, stderr = proc.communicate(timeout=4)
        assert "Traceback" not in stderr
        assert 'Fetch' in stderr
        assert proc.returncode == 0

    def test_pinecone_delete(self, index_host, mode):
        proc = spawn_locust(index_host, mode, ["--tags=delete"])
        stdout, stderr = proc.communicate(timeout=4)
        assert "Traceback" not in stderr
        assert 'Delete' in stderr
        assert proc.returncode == 0
