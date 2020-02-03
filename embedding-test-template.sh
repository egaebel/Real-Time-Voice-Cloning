#!/bin/bash

python embedding_test.py \
    --input_files_directory="../audiobook-dataset-creator/src/expanse-data/first-half-all-expanse-audio/" \
    --output_embeddings_file_path="audio-file-embeddings--file-name-to-embedding--first-half-all-expanse-audio.json" \
    --output_clusters_file_path="audio-file-embeddings--file-name-to-cluster--first-half-all-expanse-audio--spectral-clustering--2-clusters.json" \
    --clustering_method="spectral_clustering" \
    --num_clusters=2
