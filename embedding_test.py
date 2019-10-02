from encoder import inference as encoder
from encoder.params_model import model_embedding_size as speaker_embedding_size
from pathlib import Path
from sklearn.cluster import KMeans
from synthesizer.inference import Synthesizer
from time import sleep
from toolbox.utterance import Utterance
from typing import List
from utils.argutils import print_args
from vocoder import inference as vocoder

import argparse
import json
import librosa
import matplotlib
matplotlib.rcParams['backend'] = 'Qt5Agg'
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch
import umap


colormap = np.array([
    [0, 127, 70],
    [255, 0, 0],
    [255, 217, 38],
    [0, 135, 255],
    [165, 0, 165],
    [255, 167, 255],
    [97, 142, 151],
    [0, 255, 255],
    [255, 96, 38],
    [142, 76, 0],
    [33, 0, 127],
    [0, 0, 0],
    [183, 183, 183],
    [76, 255, 0],
], dtype=np.float) / 255 

min_umap_points = 4
umap_hot = False
fig = plt.figure(figsize=(4, 4), facecolor="#F0F0F0")
umap_ax = fig.subplots()
utterances = list()
file_name_to_embedding = dict()

def draw_umap_projections(utterances: List[Utterance], clusters):
    global umap_hot
    global umap_ax
    global fig
    global min_umap_points

    umap_ax.clear()

    speakers = np.unique([u.speaker_name for u in utterances])
    if clusters is not None:
        colors = {utterance.speaker_name: colormap[cluster] for i, utterance, cluster in zip(range(len(utterances)), utterances, clusters)}
    else:
        # colors = {speaker_name: colormap[i] for i, speaker_name in enumerate(speakers)}
        clusters = [0 for i, speaker_name in enumerate(speakers)]
    embeds = [u.embed for u in utterances]

    print("Utterances len: %d" % len(utterances))
    # Display a message if there aren't enough points
    if len(utterances) < min_umap_points:
        umap_ax.text(.5, .5, "Add %d more points to\ngenerate the projections" % 
                          (min_umap_points - len(utterances)), 
                          horizontalalignment='center', fontsize=15)
        umap_ax.set_title("")
        
    # Compute the projections
    else:
        if not umap_hot:
            print(
                "Drawing UMAP projections for the first time, this will take a few seconds.")
            umap_hot = True
        
        reducer = umap.UMAP(int(np.ceil(np.sqrt(len(embeds)))), metric="cosine")

        print("embeds len: %d" % len(embeds))
        # reducer = TSNE()
        projections = reducer.fit_transform(embeds)
        
        speakers_done = set()
        for projection, utterance, cluster in zip(projections, utterances, clusters):
            # color = colors[utterance.speaker_name]
            color = colormap[cluster]
            mark = "x" if "_gen_" in utterance.name else "o"
            label = None if utterance.speaker_name in speakers_done else utterance.speaker_name
            speakers_done.add(utterance.speaker_name)
            umap_ax.scatter(
                projection[0], projection[1], c=[color], marker=mark, label=label)
        # umap_ax.set_title("UMAP projections")
        umap_ax.legend(prop={'size': 10})

    # Draw the plot
    umap_ax.set_aspect("equal", "datalim")
    umap_ax.set_xticks([])
    umap_ax.set_yticks([])
    # umap_ax.figure.canvas.draw()

    if len(utterances) >= min_umap_points:
        print("Showing UMAP chart")
        plt.show()
        umap_plot_path = "umap_test.png"
        print("Saving UMAP chart at: " + umap_plot_path)
        plt.savefig(umap_plot_path)


if __name__ == '__main__':
    matplotlib.rcParams['interactive'] = True
    matplotlib.interactive(True)
    print("Matplotlib backend: %s" % str(plt.get_backend()))

    ## Info & args
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-e", "--enc_model_fpath", type=Path, 
                        default="encoder/saved_models/pretrained.pt",
                        help="Path to a saved encoder")
    parser.add_argument("-s", "--syn_model_dir", type=Path, 
                        default="synthesizer/saved_models/logs-pretrained/",
                        help="Directory containing the synthesizer model")
    parser.add_argument("-v", "--voc_model_fpath", type=Path, 
                        default="vocoder/saved_models/pretrained/pretrained.pt",
                        help="Path to a saved vocoder")
    parser.add_argument("--low_mem", action="store_true", help=\
        "If True, the memory used by the synthesizer will be freed after each use. Adds large "
        "overhead but allows to save some GPU memory for lower-end GPUs.")
    parser.add_argument("--no_sound", action="store_true", help=\
        "If True, audio won't be played.")
    args = parser.parse_args()
    print_args(args, parser)
    if not args.no_sound:
        import sounddevice as sd
        
    
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
    
    test_file_paths = [
        "/workspace-mount/Programs/audiobook-dataset-creator/src/expanse-data/the-expanse-1-leviathan-wakes/output_samples/audio/Leviathan Wakes (Unabridged) [File 01 of 57]--00001--normalized--20--00000.mp3",
        "/workspace-mount/Programs/audiobook-dataset-creator/src/expanse-data/the-expanse-1-leviathan-wakes/output_samples/audio/Leviathan Wakes (Unabridged) [File 01 of 57]--00050--normalized--20--00000.mp3",
        "/workspace-mount/Programs/audiobook-dataset-creator/src/expanse-data/the-expanse-1-leviathan-wakes/output_samples/audio/Leviathan Wakes (Unabridged) [File 01 of 57]--00051--normalized--20--00000.mp3",
        "/workspace-mount/Programs/audiobook-dataset-creator/src/expanse-data/the-expanse-1-leviathan-wakes/output_samples/audio/Leviathan Wakes (Unabridged) [File 01 of 57]--00055--normalized--20--00000.mp3",
    ]
    # test_files_dir = "/workspace-mount/Programs/audiobook-dataset-creator/src/expanse-data/the-expanse-1-leviathan-wakes/output_samples/audio/"
    test_files_dir = "/workspace-mount/Programs/audiobook-dataset-creator/src/expanse-data/the-expanse-2-calibans-war/output_samples/audio/"
    test_file_paths = sorted(map(lambda x: os.path.join(test_files_dir, x), os.listdir(test_files_dir)))
    test_file_names = list(sorted(map(lambda x: os.path.basename(x), test_file_paths)))

    print("Iterating over %d files" % len(test_file_paths))
    kmeans = KMeans(n_clusters=4, random_state=9938)
    num_generated = 0
    for test_file_path in test_file_paths:
        print("Running on path: ||%s||" % test_file_path)
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
            print("Loaded file succesfully")
        except:
            print("Error loading file: ||%s||" % test_file_path)
            continue
        
        # Then we derive the embedding. There are many functions and parameters that the 
        # speaker encoder interfaces. These are mostly for in-depth research. You will typically
        # only use this function (with its default parameters):
        embed = encoder.embed_utterance(preprocessed_wav)
        print("Created the embedding")
        
        
        ## Generating the spectrogram
        # text = input("Write a sentence (+-20 words) to be synthesized:\n")
        text = "Even the captain's cabin was indicated only by a slightly larger bunk and the face of a locked safe."
        
        # The synthesizer works in batch, so you need to put your data in a list or numpy array
        texts = [text]
        embeds = [embed]
        # If you know what the attention layer alignments are, you can retrieve them here by
        # passing return_alignments=True
        # specs = synthesizer.synthesize_spectrograms(texts, embeds)
        # spec = specs[0]
        # print("Created the mel spectrogram")
        
        
        ## Generating the waveform
        # print("Synthesizing the waveform:")
        # Synthesizing the waveform is fairly straightforward. Remember that the longer the
        # spectrogram, the more time-efficient the vocoder.
        # generated_wav = vocoder.infer_waveform(spec)
        
        
        ## Post-generation
        # There's a bug with sounddevice that makes the audio cut one second earlier, so we
        # pad it.
        # generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
        
        # Play the audio (non-blocking)
        """
        if not args.no_sound:
            sd.stop()
            sd.play(generated_wav, synthesizer.sample_rate)
        """
            
        # Add the utterance
        name = "speaker_name_placeholder_gen_%05d" % np.random.randint(100000)
        utterance = Utterance(name, "37", preprocessed_wav, None, embed, None, True)
        utterances.append(utterance)
        test_file_name = os.path.basename(test_file_path)
        file_name_to_embedding[test_file_name] = embed.tolist()
        
        # Plot it
        if (len(utterances) > 500 and len(utterances) % 500 == 0) or num_generated == (len(test_file_paths) - 1):
            with open("audio-file-embeddings--file-name-to-embedding.json", "w") as file_name_to_embedding_file:
                file_name_to_embedding_file.write(json.dumps(file_name_to_embedding))

            print("Running K-Means on %d points....." % len(utterances))
            clusters = kmeans.fit_predict(list(map(lambda x: x.embed, utterances)))
            file_name_to_cluster = dict(list(zip(test_file_names[:len(clusters)], list(map(lambda x: int(x), clusters)))))
            print("Done clustering!")

            with open("audio-file-embeddings--file-name-to-cluster.json", "w") as file_name_to_cluster_file:
                file_name_to_cluster_file.write(json.dumps(file_name_to_cluster))

            print("Drawing projections.....")
            draw_umap_projections(utterances, clusters)
            print("Done drawing!")

        # Save it on the disk
        """
        fpath = "demo_output_%02d.wav" % num_generated
        print(generated_wav.dtype)
        librosa.output.write_wav(fpath, generated_wav.astype(np.float32), 
                                 synthesizer.sample_rate)
        """
        num_generated += 1
        # print("\nSaved output as %s\n\n" % fpath)
    