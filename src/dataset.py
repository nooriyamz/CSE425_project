import os
import pandas as pd
import librosa
import numpy as np
import torch

def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    df['Audio'] = df['Audio'].str.strip()
    return df

def get_audio_file_map(audio_folder):
    return {f.lower().replace('.mp3',''): f for f in os.listdir(audio_folder)}

def get_full_path(file_map, csv_name, audio_folder):
    key = csv_name.lower()
    if key in file_map:
        return os.path.join(audio_folder, file_map[key])
    else:
        return None

def extract_mel_spectrogram(file_path, sr=22050, n_mels=64, duration=30):
    y, _ = librosa.load(file_path, sr=sr, duration=duration)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_db

def prepare_mel_tensor(df, audio_folder, max_len=1300, n_mels=64):
    file_map = get_audio_file_map(audio_folder)
    mel_specs = []

    for i, row in df.iterrows():
        audio_name = row['Audio']
        audio_file = get_full_path(file_map, audio_name, audio_folder)
        if audio_file is None:
            continue
        try:
            mel = extract_mel_spectrogram(audio_file)
        except:
            continue

        # Pad or truncate
        if mel.shape[1] < max_len:
            pad_width = max_len - mel.shape[1]
            mel = np.pad(mel, ((0,0),(0,pad_width)), mode='constant')
        elif mel.shape[1] > max_len:
            mel = mel[:, :max_len]

        # Normalize
        mel = (mel - mel.min()) / (mel.max() - mel.min())
        mel_specs.append(mel)

    mel_specs = np.array(mel_specs)
    mel_tensor = torch.tensor(mel_specs, dtype=torch.float32).unsqueeze(1)
    return mel_tensor
