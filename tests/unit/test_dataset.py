import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dataset import Dataset
import pytest


class TestDataset:

    def test_limit(self):
        limit = 123
        name = "langchain-python-docs-text-embedding-ada-002"
        dataset = Dataset(name)
        # Sanity check that the complete dataset size is greater than what
        # we are going to limit to.
        dataset_info = ([d for d in dataset.list() if d["name"] == name][0])
        assert dataset_info["documents"] > limit, \
            "Too few documents in dataset to be able to limit"

        dataset.load(limit=limit, load_queries=False)
        assert len(dataset.documents) == limit
