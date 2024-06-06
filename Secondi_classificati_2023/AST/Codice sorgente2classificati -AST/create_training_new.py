import glob
import os
import pandas as pd
import torchaudio
import torch
import torch.nn as nn
import numpy as np
from torchaudio import transforms as T
import h5py
import argparse
from args import args
import librosa
import matplotlib.pyplot as plt
import json


def load_mapping(json_file_path):
    with open(json_file_path, 'r') as json_file:
        return json.load(json_file)
    
def save_spectrogram(wav_path, output_dir):
    # Carica il file WAV
    y, sr = librosa.load(wav_path, sr=args.sr)
    
    # Calcola lo spettrogramma
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=args.nmels, fmax=8000)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    
    # Crea il nome del file di output
    base_name = os.path.splitext(os.path.basename(wav_path))[0]
    output_file = os.path.join(output_dir, f"{base_name}_spectrogram.png")
    
    # Salva lo spettrogramma come immagine
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def convert_wav_to_spectrograms(json_file_path, root_directory, output_directory):
    # Carica la mappatura dal file JSON
    mapping = load_mapping(json_file_path)
    
    # Assicurati che la cartella di output esista
    os.makedirs(output_directory, exist_ok=True)
    
    # Itera attraverso la mappatura e processa ogni file WAV
    for wav_file in mapping.keys():
        wav_path = os.path.join(root_directory, wav_file)
        if os.path.exists(wav_path):
            save_spectrogram(wav_path, output_directory)
        else:
            print(f"File non trovato: {wav_path}")

# Percorsi specifici
json_file_path = 'wav_csv_mapping.json'
root_directory = os.path.dirname(__file__)
spectrograms_folder = 'path/to/output_directory'

# Converte i file WAV in spettrogrammi
convert_wav_to_spectrograms(json_file_path, root_directory, spectrograms_folder)





traindir = args.traindir
csv_files = [f for f in glob.glob(os.path.join(traindir, '*/*.csv'))]
for csv_file in csv_files:
    ds_name = csv_file.split('/')[-2]
   

csv_filee=csv_files[1]
df = pd.read_csv(csv_filee)







