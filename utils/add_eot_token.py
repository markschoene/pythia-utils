from mmap_dataset import MMapIndexedDataset
import numpy as np
from tqdm import tqdm
import os
import argparse


path = 'pythia_pile_idxmaps/pile_0.87_deduped_text_document.bin'
print(f'Loading dataset buffer from {path}')


def add_eot_token(source_path, target_path):
    document_dataset = MMapIndexedDataset(path=source_path, skip_warmup=True)
    print(f'Dataset contains {len(document_dataset)} documents')
    raw_buffer = np.memmap(filename=source_path, mode='r', order='C', dtype=document_dataset._index.dtype)
    print(f'Dataset contains {len(raw_buffer)} tokens')

    # initialize new dataset file
    SIZE = raw_buffer.shape[0]
    SIZE_EOD_TOKENS = len(document_dataset)
    buffer_with_eod = np.memmap(target_path, shape=(SIZE + SIZE_EOD_TOKENS), mode="w+", order="C", dtype=document_dataset._index.dtype)

    counter = 0
    for doc in tqdm(document_dataset):
        start, end = counter, counter + doc.shape[0]
        buffer_with_eod[start:end] = doc
        buffer_with_eod[end] = 0
        counter += doc.shape[0] + 1

    print(f'Created dataset with EOTs at {target_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add EOT tokens to the pythia dataset"
    )
    parser.add_argument(
        "--source_file",
        type=str,
        help="Path to the unsharded .bin pile dataset",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Folder to save .bin file with EOTs into",
    )
    args = parser.parse_args()

    add_eot_token(args.source_file, args.output_file)
