#!/usr/bin/env python3

"""
Program to load a LAION dataset (https://laion.ai/blog/laion-400-open-dataset/)
into Pinecone.
"""
import pandas
from pinecone.grpc import PineconeGRPC
import pathlib
import numpy
import os
import pyarrow.parquet
import sys

from tqdm import tqdm


def split_dataframe(df, batch_size):
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i: i + batch_size]
        yield batch


def upsert_segment(index: PineconeGRPC.Index, pair):
    vectors = numpy.load(pair[0])
    metadata_rows = ["caption", "url"]
    metadata = pandas.read_parquet(pair[1], columns=["key", "shard_id", "NSFW"] + metadata_rows)

    # Sanity check - do we have the same number of records in each file?
    num_vectors = vectors.shape[0]
    num_metadata = len(metadata)
    if num_vectors != num_metadata:
        raise ValueError(f"Error: vector & metadata file pair "
                         f"({pair[0], pair[1]}) have mismatched lengths "
                         f"({num_vectors} vs {num_metadata}) - "
              "cannot process.")

    # The Pinecone SDK supports bulk loading via a Pandas DataFrame; so merge
    # the numpy vectors array into the metadata DataFrame then upsert.
    vector_series = pandas.Series(list(vectors))
    df = metadata.assign(values = vector_series)

    # Remove any records which have an 'NSFW' which is not 'UNLIKELY'
    df = df[df.NSFW == "UNLIKELY"]

    # Reformat the DataFrame for upsert_from_dataframe()
    # - Combine any metadate columns (we want to keep) into a single 'metadata'
    #   dictionary. Note we crop any individual metadata field to 10,000 B
    #   to ensure we keep to the overall Pinecone metadata limit of 40,000 B.
    def merge_metadata(row):
        return {m: row[m][:10000] for m in metadata_rows}
    df["metadata"] = df.apply(merge_metadata, axis=1)

    # - Combine shard_id and key into a unique 'id' field.
    def make_id(row):
        return str(row['shard_id']) + "-" + str(row['key'])
    df["id"] = df.apply(make_id, axis=1)

    # - drop any columns it doesn't need / expect.
    df = df.drop(columns=["key", "shard_id", "NSFW"] + metadata_rows)

    # For large dataframes (like the ~1,000,000 records we have here)
    # dispatching a large number of async requests (1,000,000 / batch size)
    # and then waiting for them all to complete can result in timeouts.
    # Instead, dispatch in smaller chunks.
    pbar = tqdm(desc="Upserting vectors", unit=" vectors", total=len(df))
    upserted_count = 0

    for sub_frame in split_dataframe(df, 10000):
        resp = index.upsert_from_dataframe(sub_frame,
                                           batch_size=200,
                                           show_progress=False)
        upserted_count += resp.upserted_count
        pbar.update(len(sub_frame))

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <path/to/laion/dataset> <Pinecone_URL>",
              file=sys.stderr)
        return 1

    dataset = pathlib.Path(sys.argv[1])
    index_host = sys.argv[2]
    api_key = os.environ.get('PINECONE_API_KEY', None)
    if not api_key:
        print("Error: PINECONE_API_KEY environment variable not set or empty."
              + " Set to the API key to connect to Pinecone with.",
              file=sys.stderr)
        return 1

    print(f"Loading dataset '{dataset.name}' into Pinecone index '{index_host}'...")

    # Check the directory layout is as expected, and scan the images and
    # metadata files.
    files = list()
    for subdir, pattern in [("images", "*.npy"), ("metadata", "*.parquet")]:
        path = (dataset / subdir)
        if not path.exists():
            print(f"Error: {dataset.name} is missing expected subdirectory "
                  f"'{subdir}'. Check dataset is correct.",
                  file=sys.stderr)
            return 1
        if len([path.glob(pattern)]) == 0:
            print(f"Error: {dataset.name} does not contain any files matching "
                  f"pattern '{pattern}'. Expected at least one file.",
                  file=sys.stderr)
            return 1
        files.append(list(path.glob(pattern)))

    # Check the number of image and metadata files are equal.
    if len(files[0]) != len(files[1]):
        print(f"Error: Number of images files ({len(files[0])} is not equal "
              f"to the number of metadata files ({len(files[1])}). Expected "
              "an equal number.")

    # Check the index exists and we can connect to it. Using GRPC as for
    # bulk upsert it is quicker than HTTP.
    pc = PineconeGRPC(api_key=api_key)
    index = pc.Index(host=index_host)

    # Iterate across all pairs of (vector, metadata) files
    for pair in tqdm(zip(files[0], files[1]), desc="Processing segment",
                     total=len(list(files[0]))):
        upsert_segment(index, pair)


if __name__ == "__main__":
    main()
