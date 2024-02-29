from google.cloud.storage import Bucket, Client, transfer_manager
import json
import logging
import pandas
import pathlib
from pinecone.grpc import PineconeGRPC
from pyarrow.parquet import ParquetDataset
from tqdm import tqdm, trange


class Dataset:
    """
    Represents a Dataset used as the source of documents and/or queries for
    Pinecone index operations.
    The set of datasets are taken from the Pinecone public datasets
    (https://docs.pinecone.io/docs/using-public-datasets), which reside in a
    Google Cloud Storage bucket and are downloaded on-demand on first access,
    then cached on the local machine.
    """
    gcs_bucket = "pinecone-datasets-dev"

    @staticmethod
    def split_dataframe(df, batch_size):
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i: i + batch_size]
            yield batch

    def __init__(self, name: str = "", cache_dir: str = ""):
        self.name = name
        self.cache = pathlib.Path(cache_dir)
        self.documents = None
        self.queries = pandas.DataFrame()

    @staticmethod
    def list():
        """
        List all available datasets on the GCS bucket.
        :return: A list of dict objects, one for each dataset.
        """
        client = Client.create_anonymous_client()
        bucket: Bucket = client.bucket(Dataset.gcs_bucket)
        metadata_blobs = bucket.list_blobs(match_glob="*/metadata.json")
        datasets = []
        for m in metadata_blobs:
            datasets.append(json.loads(m.download_as_string()))
        return datasets

    def load(self, skip_download: bool = False, load_queries: bool = True,
             doc_sample_fraction: float = 1.0):
        """
        Load the dataset, populating the 'documents' and 'queries' DataFrames.
        """
        if not skip_download:
            self._download_dataset_files()

        # Load all the parquet dataset (made up of one or more parquet files),
        # to use for documents into a pandas dataframe.
        self.documents = self._load_parquet_dataset("documents")

        # If there is an explicit 'queries' dataset, then load that and use
        # for querying, otherwise use documents directly.
        if load_queries:
            self.queries = self._load_parquet_dataset("queries")
        if not self.queries.empty:
            logging.info(
                f"Using {len(self.queries)} query vectors loaded from dataset 'queries' table")
        else:
            # Queries expect a different schema than documents.
            # Documents looks like:
            #    ["id", "values", "sparse_values", "metadata"]
            # Queries looks like:
            #    ["vector", "sparse_vector", "filter", "top_k"]
            #
            # Extract 'values' and rename to query schema (only
            # 'vector' field of queries is currently used).
            self.queries = self.documents[["values"]].copy()
            self.queries.rename(columns={"values": "vector"}, inplace=True)

            # Use a sampling of documents for queries (to avoid
            # keeping a large complete dataset in memory for each
            # worker process).
            self.queries = self.queries.sample(frac=doc_sample_fraction, random_state=1)
            logging.info(
                f"Using {doc_sample_fraction * 100}% of documents' dataset "
                f"for query data ({len(self.queries)} sampled")

            # Convert the vectors from pandas DataFrame to Python objects
            # up-front as it's expensive to iterate on DataFrame (query
            # sampling must iterate).
            self.queries = self.queries.astype(dtype={'vector': object})

    def upsert_into_index(self, index_host, api_key, skip_if_count_identical: bool = False):
        """
        Upsert the datasets' documents into the specified index.
        :param index_host: Pinecone index to upsert into (must already exist)
        :param skip_if_count_identical: If true then skip upsert if the index already contains the same number of
               vectors as the dataset.
        """
        pinecone = PineconeGRPC(api_key)
        index = pinecone.Index(host=index_host)
        if skip_if_count_identical:
            if index.describe_index_stats()['total_vector_count'] == len(self.documents):
                logging.info(
                    f"Skipping upsert as index already has same number of documents as dataset ({len(self.documents)}")
                return

        upserted_count = self._upsert_from_dataframe(index)
        if upserted_count != len(self.documents):
            logging.warning(
                f"Not all records upserted successfully. Dataset count:{len(self.documents)},"
                f" upserted count:{upserted_count}")

    def prune_documents(self):
        """
        Discard the contents of self.documents once it is no longer required
        (it can consume a significant amount of memory).
        """
        del self.documents
        logging.debug(f"After pruning, 'queries' memory usage:{self.queries.memory_usage()}")

    def _download_dataset_files(self):
        self.cache.mkdir(parents=True, exist_ok=True)
        logging.debug(f"Checking for existence of dataset '{self.name}' in dataset cache '{self.cache}'")
        client = Client.create_anonymous_client()
        bucket: Bucket = client.bucket(Dataset.gcs_bucket)
        blobs = [b for b in bucket.list_blobs(prefix=self.name + "/")]
        # Ignore directories (blobs ending in '/') as we don't explicilty need them
        # (non-empty directories will have their files downloaded
        # anyway).
        blobs = [b for b in blobs if not b.name.endswith("/")]
        logging.debug(f"Dataset consists of files:{[b.name for b in blobs]}")

        def should_download(blob):
            path = self.cache / blob.name
            if not path.exists():
                return True
            # File exists - check size, assume same size is same file.
            # (Ideally would check hash (md5), but using hashlib.md5() to
            # calculate the local MD5 does not match remove; maybe due to
            # transmission as compressed file?
            local_size = path.stat().st_size
            remote_size = blob.size
            return local_size != remote_size

        to_download = [b for b in filter(lambda b: should_download(b), blobs)]
        if not to_download:
            return
        pbar = tqdm(desc="Downloading datset",
                    total=sum([b.size for b in to_download]),
                    unit="Bytes",
                    unit_scale=True)
        for blob in to_download:
            logging.debug(f"Dataset file '{blob.name}' not found in cache - will be downloaded")
            dest_path = self.cache / blob.name
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(self.cache / blob.name)
            pbar.update(blob.size)

    def _load_parquet_dataset(self, kind):
        parquet_files = [f for f in (self.cache / self.name).glob(kind + '/*.parquet')]
        if not len(parquet_files):
            return pandas.DataFrame

        dataset = ParquetDataset(parquet_files)
        # Read only the columns that Pinecone SDK makes use of.
        if kind == "documents":
            columns = ["id", "values", "sparse_values", "metadata"]
            metadata_column = "metadata"
        elif kind == "queries":
            columns = ["vector", "sparse_vector",  "filter", "top_k", "blob"]
            metadata_column = "filter"
        else:
            raise ValueError(f"Unsupported kind '{kind}' - must be one of (documents, queries)")
        # Note: We to specify pandas.ArrowDtype as the types mapper to use pyarrow datatypes in the
        # resulting DataFrame. This is significant as (for reasons unknown) it allows subsequent
        # samples() of the DataFrame to be "disconnected" from the original underlying pyarrow data,
        # and hence significantly reduces memory usage when we later prune away the underlying
        # parrow data (see prune_documents).
        df = dataset.read(columns=columns).to_pandas(types_mapper=pandas.ArrowDtype)

        # And drop any columns which all values are missing - e.g. not all
        # datasets have sparse_values, but the parquet file may still have
        # the (empty) column present.
        df.dropna(axis='columns', how="all", inplace=True)

        if metadata_column in df:
            def cleanup_null_values(metadata):
                # Null metadata values are not supported, remove any key
                # will a null value.
                if not metadata:
                    return None
                return {k: v for k, v in metadata.items() if v}

            def convert_metadata_to_dict(metadata) -> dict:
                # metadata is expected to be a dictionary of key-value pairs;
                # however it may be encoded as a JSON string in which case we
                # need to convert it.
                if metadata is None:
                    return None
                if isinstance(metadata, dict):
                    return metadata
                if isinstance(metadata, str):
                    return json.loads(metadata)
                raise TypeError(f"metadata must be a string or dict (found {type(metadata)})")

            def prepare_metadata(metadata):
                return cleanup_null_values(convert_metadata_to_dict(metadata))

            df[metadata_column] = df[metadata_column].apply(prepare_metadata)
        logging.debug(f"Loaded {len(df)} vectors of kind '{kind}'")
        return df

    def _upsert_from_dataframe(self, index):
        """
        Note: using PineconeGRPC.Index.upsert_from_dataframe() directly
        results in intermittent failures against serverless indexes as
        we can hit the request limit:
            grpc._channel._MultiThreadedRendezvous: < _MultiThreadedRendezvous of RPC that terminated with:
               status = StatusCode.RESOURCE_EXHAUSTED
               details = "Too many requests. Please retry shortly"
        I haven't observed this with the HTTP Pinecone.Index, however the
        gRPC one is so much faster for bulk loads we really want to keep using
        gRPC. As such, we have our own version of upsert from dataframe which
        handles this error with backoff and retry.
        """

        # Solution is somewhat naive - simply chunk the dataframe into
        # chunks of a smaller size, and pass each chunk to upsert_from_dataframe.
        # We still end up with multiple vectors in progress at once, but we
        # limit it to a finite amount and not the entire dataset.
        pbar = tqdm(desc="Populating index", unit=" vectors", total=len(self.documents))
        upserted_count = 0

        for sub_frame in Dataset.split_dataframe(self.documents, 10000):
            # The 'values' column in the DataFrame is a pyarrow type (list<item: double>[pyarrow])
            # as it was read using the pandas.ArrowDtype types_mapper (see _load_parquet_dataset).
            # This _can_ be automatically converted to a Python list object inside upsert_from_dataframe,
            # but it is slow, as at that level the DataFrame is iterated row-by-row and the conversion
            # happens one element at a time.
            # However, converting the entire sub-frame's column back to a Python object before calling
            # upsert_from_dataframe() is significantly faster, such that the overall upsert throughput
            # (including the actual server-side work) is around 2x greater if we pre-convert.
            converted = sub_frame.astype(dtype={'values': object})
            resp = index.upsert_from_dataframe(converted,
                                               batch_size=200,
                                               show_progress=False)
            upserted_count += resp.upserted_count
            pbar.update(len(sub_frame))
        return upserted_count
