import os
import h5py
import pandas as pd
import torchaudio
import torch
import args
from torch.utils.data import Dataset, DataLoader
from torchaudio import transforms as T
import glob
# Definizione delle variabili
N_MELS = args.nmels
TARGET_SR = args.sr  #sampling rate, di default 22050
N_FFT = args.nfft  #size della Fast Fourier Transform
N_MELS = args.nmels #numero di mels.
HOP_MEL = args.hoplen #hop between STFT windows
FMIN = args.fmin
FMAX = args.fmax
N_SHOT = args.nshot #numero di shot pari a 5.
fps = TARGET_SR/HOP_MEL 
WIN_LEN = args.len
SEG_LEN = WIN_LEN//2
win_len = int(round((WIN_LEN/1000) * fps))
seg_hop = int(round((SEG_LEN/1000) * fps))


fps = 30
TARGET_SR = 16000
BATCH_SIZE = 32
map_cls_2_int = {'class1': 0, 'class2': 1, 'class3': 2}  
transform = T.MelSpectrogram(sample_rate=TARGET_SR, n_mels=N_MELS)

# Percorso del file HDF5
traindir = args.train_dir


csv_files = [f for f in glob.glob(os.path.join(traindir, '*/*.csv'))]
hdf_tr = os.path.join(traindir, 'train.h5')
hf = h5py.File(hdf_tr, 'w')
# Crea il file H5
hf.create_dataset('data', shape=(0, N_MELS, win_len), maxshape=(None, N_MELS, win_len))
hf.create_dataset('label', shape=(0, 1), maxshape=(None, 1))

# Estrai dati dal file HDF5
if len(hf['data'][:]) == 0:
    file_index = 0
else:
    file_index = len(hf['data'][:])

for csv_file in csv_files:
    counter = 0
    ds_name = csv_file.split('/')[-2]
    wav_file = csv_file.replace('csv', 'wav')
    print(f"Creating data for {wav_file}")
    df = pd.read_csv(csv_file)
    df['Starttime'] = df['Starttime']
    df['Endtime'] = df['Endtime']
    wav, sr = torchaudio.load(wav_file)
    resample = T.Resample(sr, TARGET_SR)
    wav = resample(wav)
    if wav.shape[0] != 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    melspec = transform(wav)

    df_cols = df.columns.tolist()
    dfs = [df[df[col] == 'POS'] for col in df_cols if len(df[df[col] == 'POS']) > 0]
    dfs = pd.concat(dfs)

    for i in range(len(dfs)):
        ith_row = dfs.iloc[i]
        for df_col in df_cols:
            if ith_row[df_col] == 'POS':
                label = df_col
                break
        onset = int(round(ith_row['Starttime'] * fps))
        offset = int(round(ith_row['Endtime'] * fps))
        start_idx = onset

        if offset - start_idx > win_len:
            while offset - start_idx > win_len:
                spec = melspec[..., start_idx:start_idx + win_len]
                if spec.sum() == 0:
                    counter += 1
                    start_idx += seg_hop
                    continue
                spec = (spec - spec.min()) / (spec.max() - spec.min())
                hf['data'].resize((file_index + 1, spec.shape[-2], spec.shape[-1]))
                hf['data'][file_index] = spec
                hf['label'].resize((file_index + 1, 1))
                hf['label'][file_index] = map_cls_2_int[label]
                file_index += 1
                start_idx += seg_hop

            if offset - start_idx > win_len // 8:
                spec = melspec[..., start_idx:offset]
                repeat_num = int(win_len / spec.shape[-1]) + 1
                spec = spec.repeat(1, 1, repeat_num)
                spec = spec[..., :win_len]
                spec = (spec - spec.min()) / (spec.max() - spec.min())
                hf['data'].resize((file_index + 1, spec.shape[-2], spec.shape[-1]))
                hf['data'][file_index] = spec
                hf['label'].resize((file_index + 1, 1))
                hf['label'][file_index] = map_cls_2_int[label]
                file_index += 1
        else:
            if offset - start_idx > win_len // 8:
                spec = melspec[..., start_idx:offset]
                if spec.sum() == 0:
                    counter += 1
                    continue
                repeat_num = int(win_len / spec.shape[-1]) + 1
                spec = spec.repeat(1, 1, repeat_num)
                spec = spec[..., :win_len]
                spec = (spec - spec.min()) / (spec.max() - spec.min())
                hf['data'].resize((file_index + 1, spec.shape[-2], spec.shape[-1]))
                hf['data'][file_index] = spec
                hf['label'].resize((file_index + 1, 1))
                hf['label'][file_index] = map_cls_2_int[label]
                file_index += 1

    if counter > 0:
        print(f"{counter} patches are null in {wav_file}")
print(f"Total files created: {file_index}")
hf.close()

# Classe Dataset per AST
class SpectrogramDataset(Dataset):
    def __init__(self, hdf5_file):
        self.hdf5_file = hdf5_file
        with h5py.File(hdf5_file, 'r') as hdf:
            self.data = hdf['data'][:]
            self.labels = hdf['label'][:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        spectrogram = self.data[idx]
        label = self.labels[idx]
        return torch.tensor(spectrogram, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# Creazione del DataLoader
dataset = SpectrogramDataset(hdf_tr)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# Esempio di utilizzo del DataLoader
for spectrograms, labels in dataloader:
    # Addestramento del modello AST qui
    # Modello AST si aspetta input di dimensioni [batch_size, num_channels, freq_bins, time_steps]
    # Assumendo che il modello AST abbia un metodo `forward` che prende in input spectrograms
    output = ast_model(spectrograms)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()
