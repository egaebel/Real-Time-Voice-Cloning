from collections import Counter, defaultdict
from datetime import timedelta
from encoder import inference as encoder
from encoder.params_model import model_embedding_size as speaker_embedding_size
from pathlib import Path
from sklearn.cluster import AffinityPropagation, DBSCAN, KMeans, MeanShift, SpectralClustering, estimate_bandwidth
from synthesizer.inference import Synthesizer
from time import sleep
from timeit import default_timer as timer
from typing import List
from utils.argutils import print_args
from vocoder import inference as vocoder

import argparse
import itertools
import json
import librosa
import math
import matplotlib
matplotlib.rcParams['backend'] = 'Qt5Agg'
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys
import torch
import umap

CLUSTER_AND_PLOT_ITERATIONS = 2000

colormap = np.array([
    [255,0,0], 
    [255,145,128], 
    [153,41,0], 
    [89,57,45], 
    [179,146,134], 
    [255,102,0], 
    [255,179,128], 
    [178,95,0], 
    [64,34,0], 
    [140,108,70], 
    [255,170,0], 
    [255,234,191], 
    [64,58,48], 
    [89,71,0], 
    [178,152,45], 
    [255,238,0], 
    [140,138,105], 
    [242,255,191], 
    [170,255,0], 
    [156,204,102], 
    [88,115,57], 
    [0,255,102], 
    [0,166,88], 
    [0,77,51], 
    [172,230,210], 
    [0,255,204], 
    [0,140,131], 
    [0,238,255], 
    [0,163,204], 
    [0,82,102], 
    [0,170,255], 
    [191,234,255], 
    [0,95,179], 
    [0,41,77], 
    [0,102,255], 
    [105,119,140], 
    [0,20,77], 
    [191,208,255], 
    [0,34,255], 
    [0,20,153], 
    [89,70,140], 
    [179,128,255], 
    [217,191,255], 
    [204,0,255], 
    [122,0,153], 
    [61,0,77], 
    [255,0,204], 
    [191,96,159], 
    [115,86,105], 
    [255,0,136], 
    [140,0,75], 
    [64,0,34], 
    [255,191,225], 
    [255,0,68], 
    [115,0,15], 
    [191,96,108], 
    [76,38,43],
], dtype=np.float) / 255 

min_umap_points = 4
umap_hot = False
fig = plt.figure(figsize=(4, 4), facecolor="#F0F0F0")
umap_ax = fig.subplots()
embeddings = list()
file_name_to_embedding = dict()

def sample_points(embeddings, clusters, percentage_per_cluster):
    cluster_to_embeddings = defaultdict(list)
    for cluster, embedding in zip(clusters, embeddings):
        cluster_to_embeddings[cluster].append(embedding)
    for cluster in cluster_to_embeddings.keys():
        embeddings_to_sample = cluster_to_embeddings[cluster]
        target_size = math.ceil(percentage_per_cluster * len(embeddings_to_sample))
        random.seed()
        while len(embeddings_to_sample) > target_size:
            index_to_delete = random.randint(0, len(embeddings_to_sample) - 1)
            embeddings_to_sample.pop(index_to_delete)
    sampled_embeddings = list()
    sampled_clusters = list()
    for sampled_cluster, sampled_embedding in itertools.chain.from_iterable(map(lambda x: list(map(lambda y: (x[0], y), x[1])), cluster_to_embeddings.items())):
        sampled_embeddings.append(sampled_embedding)
        sampled_clusters.append(sampled_cluster)
    return sampled_embeddings, sampled_clusters


def draw_umap_projections(embeddings, clusters, sampling_percentage=None):
    global umap_hot
    global umap_ax
    global fig
    global min_umap_points

    umap_ax.clear()

    assert len(embeddings) == len(clusters)

    print("draw_umap_projections called with %d points" % len(embeddings))
    if sampling_percentage is not None:
        print("Sampling %f points....." % sampling_percentage)
        embeddings, clusters = sample_points(embeddings, clusters, sampling_percentage)
        print("Sampled down to %d points!" % len(embeddings))

    print("Utterances len: %d" % len(embeddings))
    # Display a message if there aren't enough points
    if len(embeddings) < min_umap_points:
        umap_ax.text(.5, .5, "Add %d more points to\ngenerate the projections" % 
                          (min_umap_points - len(embeddings)), 
                          horizontalalignment='center', fontsize=15)
        umap_ax.set_title("")
        
    # Compute the projections
    else:
        if not umap_hot:
            print(
                "Drawing UMAP projections for the first time, this will take a few seconds.")
            umap_hot = True
        
        print("Computing projections.....")
        reducer = umap.UMAP(int(np.ceil(np.sqrt(len(embeddings)))), metric="cosine")
        print("embeddings len: %d" % len(embeddings))
        # reducer = TSNE()
        projections = reducer.fit_transform(embeddings)
        print("Computed projections!")
        
        print("Plotting projections.....")
        projection_count = 0
        for projection, embedding, cluster in zip(projections, embeddings, clusters):
            color = colormap[cluster]
            mark = "x"
            umap_ax.scatter(
                projection[0], projection[1], c=[color], marker=mark, label=cluster)
            projection_count += 1
        # umap_ax.set_title("UMAP projections")
        umap_ax.legend(prop={'size': 10})

    # Draw the plot
    umap_ax.set_aspect("equal", "datalim")
    umap_ax.set_xticks([])
    umap_ax.set_yticks([])
    print("Plotted projections!")
    print("Remember, there are %d clusters!" % len(set(clusters)))

    if len(embeddings) >= min_umap_points:
        print("Showing UMAP chart")
        plt.show()
        umap_plot_path = "umap_test.png"
        print("Saving UMAP chart at: " + umap_plot_path)
        plt.savefig(umap_plot_path)


def get_clustering_algorithm(clustering_algorithm, x, num_clusters=4):
    clustering_algorithms = {
        "k_means": KMeans(n_clusters=int(num_clusters), random_state=9938),
        "spectral_clustering": SpectralClustering(n_clusters=int(num_clusters)),
        "mean_shift": MeanShift(bandwidth=estimate_bandwidth(x, quantile=0.2, n_samples=int(0.1 * len(x)))),
        "affinity_propagation": AffinityPropagation(damping=0.95, preference=50),
        "dbscan": DBSCAN(eps=0.5, metric="l1", min_samples=5),
    }
    print("Using clustering algorithm: %s" % clustering_algorithm)
    if clustering_algorithm in ("k_means", "spectral_clustering"):
        print("Using num_clusters: %s" % str(num_clusters))
    return clustering_algorithms[clustering_algorithm]

def save_and_cluster(args, test_file_names, embeddings, file_name_to_embedding):
    print("Writing embedding file.....")
    with open(args.output_embeddings_file_path, "w") as file_name_to_embedding_file:
        file_name_to_embedding_file.write(json.dumps(file_name_to_embedding))

    print("Running clustering on %d points....." % len(embeddings))
    clustering_start = timer()
    clusters = get_clustering_algorithm(args.clustering_method, embeddings, args.num_clusters).fit_predict(embeddings)
    clusters_int = list(map(lambda x: int(x), clusters))
    file_name_to_cluster = dict(list(zip(test_file_names[:len(clusters)], clusters_int)))
    clustering_end = timer()
    print("Done clustering!")
    print("Produced %d clusters!" % len(set(clusters_int)))
    print("Clusters have counts: ")
    print(sorted(Counter(clusters_int).items(), key=lambda x: -x[1]))
    print("Clustering took: %s" % str(timedelta(seconds=clustering_end - clustering_start)))

    print("Writing clusters file.....")
    with open(args.output_clusters_file_path, "w") as file_name_to_cluster_file:
        file_name_to_cluster_file.write(json.dumps(file_name_to_cluster))

    if args.create_projection:
        print("Drawing projections.....")
        projection_start = timer()
        draw_umap_projections(embeddings, clusters, sampling_percentage=args.sample_from_clusters)
        projection_end = timer()
        print("Done drawing!")
        print("Projection took: %s" % str(timedelta(seconds=projection_end - projection_start)))

if __name__ == '__main__':
    matplotlib.rcParams['interactive'] = True
    matplotlib.interactive(True)
    print("Matplotlib backend: %s" % str(plt.get_backend()))

    ## Info & args
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-e",
                        "--enc_model_fpath", 
                        type=Path, 
                        default="encoder/saved_models/pretrained.pt",
                        help="Path to a saved encoder")
    parser.add_argument("--file_name_to_embedding_file_path", type=Path, default=None, help=\
        "Path to a file name to audio embedding dictionary saved in JSON.")
    parser.add_argument("--file_name_to_cluster_file_path", type=Path, default=None, help=\
        "Path to a file name to cluster dictionary saved in JSON.")
    parser.add_argument("-c", "--create_projection", action="store_true", help=\
        "If true create a umap or tsne projection.")
    parser.add_argument("--output_embeddings_file_path", type=Path, default="audio-file-embeddings--file-name-to-embedding.json", help=\
        "Output path indicating where the mapping from file names to embeddings should be stored.")
    parser.add_argument("--output_clusters_file_path", type=Path, default="audio-file-embeddings--file-name-to-cluster.json", help=\
        "Output path indicating where the mapping from file names to cluster number should be stored.")
    parser.add_argument("--sample_from_clusters", default=0.2, help=\
        "Specifies that we should sample from clusters for plotting at the percentage specified.")        
    parser.add_argument("--clustering_method", default="k_means", help="Which algorithm to use for clustering embeddings.")
    parser.add_argument("--num_clusters", default=4, help="For algorithms that take cluster number, the number of clusters to produce.")
    parser.add_argument("--input_files_directory",
        default=None,
        help="A directory full of audio files that should have embeddings created for them (I think they have to be mp3s.....).")
    args = parser.parse_args()
    print_args(args, parser)
        
    
    ## Print some environment information (for debugging purposes)
    print("Running a test of your configuration...\n")
    if not torch.cuda.is_available():
        print("Your PyTorch installation is not configured to use CUDA. If you have a GPU ready "
              "for deep learning, ensure that the drivers are properly installed, and that your "
              "CUDA version matches your PyTorch installation. CPU-only inference is currently "
              "not supported.", file=sys.stderr)
        quit(-1)
    device_id = torch.cuda.current_device()
    gpu_properties = torch.cuda.get_device_properties(device_id)
    print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
          "%.1fGb total memory.\n" % 
          (torch.cuda.device_count(),
           device_id,
           gpu_properties.name,
           gpu_properties.major,
           gpu_properties.minor,
           gpu_properties.total_memory / 1e9))
    
    
    ## Load the models one by one.
    print("Preparing the encoder, the synthesizer and the vocoder...")
    encoder.load_model(args.enc_model_fpath)

    # Embeddings and clusters have been generated already, only create the projection.
    if (args.file_name_to_cluster_file_path is not None
            and args.file_name_to_embedding_file_path is not None
            and args.create_projection):
        print("Using file_name_to_cluster_file_path: ||%s||" % args.file_name_to_cluster_file_path)
        print("Using file_name_to_embedding_file_path: ||%s||" % args.file_name_to_embedding_file_path)
        with open(args.file_name_to_embedding_file_path, "r") as file_name_to_embedding_file:
            file_name_to_embedding_dict = json.loads(file_name_to_embedding_file.read())
        with open(args.file_name_to_cluster_file_path, "r") as file_name_to_cluster_file:
            file_name_to_cluster_dict = json.loads(file_name_to_cluster_file.read())
        assert len(file_name_to_embedding_dict) == len(file_name_to_cluster_dict), "len(file_name_to_embedding_dict) = %d while len(file_name_to_cluster_dict) = %d" % (len(file_name_to_embedding_dict), len(file_name_to_cluster_dict))
        assert file_name_to_embedding_dict.keys() == file_name_to_cluster_dict.keys(), "file_name_to_embedding_dict.keys() != file_name_to_cluster_dict.keys(), left out keys are: %s" % (set(file_name_to_embedding_dict.keys()).symmetric_difference(set(file_name_to_cluster_dict.keys())))
        file_name_cluster_embedding_tuple_list = [(key, file_name_to_cluster_dict[key], file_name_to_embedding_dict[key]) for key in file_name_to_embedding_dict.keys()]

        print("Drawing projections.....")
        draw_umap_projections(
            list(map(lambda x: x[2], file_name_cluster_embedding_tuple_list)), 
            list(map(lambda x: x[1], file_name_cluster_embedding_tuple_list)),
            sampling_percentage=args.sample_from_clusters)
        print("Drew projections!")
        sys.exit(0)

    # The embedding has already been generated, only perform clustering.
    if args.file_name_to_embedding_file_path is not None:
        print("Using file_name_to_embedding_file_path: ||%s||" % args.file_name_to_embedding_file_path)
        with open(args.file_name_to_embedding_file_path, "r") as file_name_to_embedding_file:
            file_name_to_embedding_dict = json.loads(file_name_to_embedding_file.read())
            file_name_to_embedding_tuple_list = list(file_name_to_embedding_dict.items())

        print("Running clustering.....")
        loaded_embeddings = list(map(lambda x: x[1], file_name_to_embedding_tuple_list))
        clustering_start = timer()
        clusters = get_clustering_algorithm(args.clustering_method, loaded_embeddings, args.num_clusters).fit_predict(loaded_embeddings)
        clustering_end = timer()
        clusters_int = list(map(lambda x: int(x), clusters))
        file_name_to_cluster = dict(list(zip(list(map(lambda x: x[0], file_name_to_embedding_tuple_list)), clusters_int)))
        with open(args.output_clusters_file_path, "w") as file_name_to_cluster_file:
            file_name_to_cluster_file.write(json.dumps(file_name_to_cluster))
        print("Clustering took: %s" % str(timedelta(seconds=clustering_end - clustering_start)))
        print("Produced %d clusters!" % len(set(clusters_int)))
        print("Clusters have counts: ")
        print(sorted(Counter(clusters_int).items(), key=lambda x: -x[1]))
        print("Wrote clusters file.")
        if args.create_projection:
            print("Drawing projections.....")
            draw_umap_projections(
                loaded_embeddings,
                clusters,
                sampling_percentage=args.sample_from_clusters)
            print("Drew projections!")
        sys.exit(0)
    
    """
    test_file_paths = [
        "/home/egaebel/workspace/Programs/audiobook-dataset-creator/src/expanse-data/the-expanse-1-leviathan-wakes/output_samples/audio/Leviathan Wakes (Unabridged) [File 01 of 57]--00001--normalized--20--00000.mp3",
        "/home/egaebel/workspace/Programs/audiobook-dataset-creator/src/expanse-data/the-expanse-1-leviathan-wakes/output_samples/audio/Leviathan Wakes (Unabridged) [File 01 of 57]--00050--normalized--20--00000.mp3",
        "/home/egaebel/workspace/Programs/audiobook-dataset-creator/src/expanse-data/the-expanse-1-leviathan-wakes/output_samples/audio/Leviathan Wakes (Unabridged) [File 01 of 57]--00051--normalized--20--00000.mp3",
        "/home/egaebel/workspace/Programs/audiobook-dataset-creator/src/expanse-data/the-expanse-1-leviathan-wakes/output_samples/audio/Leviathan Wakes (Unabridged) [File 01 of 57]--00055--normalized--20--00000.mp3",
    ]
    """
    # test_files_dir = "/workspace/Programs/audiobook-dataset-creator/src/expanse-data/the-expanse-1-leviathan-wakes/output_samples/audio/"
    # test_files_dir = "/workspace/Programs/audiobook-dataset-creator/src/expanse-data/the-expanse-2-calibans-war/output_samples/audio/"
    # test_files_dir = "/home/egaebel/workspace/Programs/audiobook-dataset-creator/src/expanse-data/all-expanse-audio"
    test_files_dir = args.input_files_directory
    test_file_paths = sorted(map(lambda x: os.path.join(test_files_dir, x), os.listdir(test_files_dir)))
    test_file_names = list(sorted(map(lambda x: os.path.basename(x), test_file_paths)))

    print("Iterating over files in dir: %s" % test_files_dir)
    print("Iterating over %d files" % len(test_file_paths))
    num_generated = 0
    for test_file_path in test_file_paths:
        # print("Running on path: ||%s||" % test_file_path)
        ## Computing the embedding
        # First, we load the wav using the function that the speaker encoder provides. This is 
        # important: there is preprocessing that must be applied.
        
        try:
            # The following two methods are equivalent:
            # - Directly load from the filepath:
            preprocessed_wav = encoder.preprocess_wav(test_file_path)
            # - If the wav is already loaded:
            original_wav, sampling_rate = librosa.load(test_file_path)
            preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
            # print("Loaded file succesfully")
        except:
            print("Error loading file: ||%s||" % test_file_path)
            continue
        
        # Then we derive the embedding. There are many functions and parameters that the 
        # speaker encoder interfaces. These are mostly for in-depth research. You will typically
        # only use this function (with its default parameters):
        embed = encoder.embed_utterance(preprocessed_wav)
        # print("Created the embedding")
        
        embeds = [embed]
            
        # Add the utterance
        name = "speaker_name_placeholder_gen_%05d" % np.random.randint(100000)
        embeddings.append(embed)
        test_file_name = os.path.basename(test_file_path)
        file_name_to_embedding[test_file_name] = embed.tolist()
        
        num_generated += 1

        # Plot it
        if ((len(embeddings) > CLUSTER_AND_PLOT_ITERATIONS and len(embeddings) % CLUSTER_AND_PLOT_ITERATIONS == 0)
                or num_generated == len(test_file_paths)):
            save_and_cluster(args, test_file_names, embeddings, file_name_to_embedding)

    
        # print("\nSaved output as %s\n\n" % fpath)
    print("Writing final embeddings and clusters.....")
    save_and_cluster(args, test_file_names, embeddings, file_name_to_embedding)
    print("Done!")