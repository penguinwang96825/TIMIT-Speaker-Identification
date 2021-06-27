import os
import re
import librosa
import numpy as np
import pandas as pd
from itertools import starmap
from datasets import load_dataset


def get_timit_metadata(timit_dir):
    """
    Parameters
    ----------
    timit_dir: str
        Path directory for TRAIN or TEST data.
    """
    timit_data = []
    for dialect_region in os.listdir(timit_dir):
        dialect_region_dir_path = os.path.join(timit_dir, dialect_region)
        for speaker_id in os.listdir(dialect_region_dir_path):
            speaker_id_dir_path = os.path.join(dialect_region_dir_path, speaker_id)
            for file in os.listdir(speaker_id_dir_path):
                if file.endswith("WAV"):
                    id_ = file.split(".")[0]
                    sentence_type = re.findall("[A-Za-z]+", id_.strip())[0]
                    file_path = os.path.join(speaker_id_dir_path, file)
                    timit_data.append([
                        dialect_region, 
                        file_path, 
                        id_, 
                        sentence_type, 
                        speaker_id
                    ])
    timit_data = pd.DataFrame(
        timit_data, 
        columns=["dialect_region", "file", "id", "sentence_type", "speaker_id"]
    )
    timit_data = timit_data.sort_values("speaker_id").reset_index(drop=True)
    return timit_data


def get_timit_metadata_from_huggingface():
    timit = load_dataset('timit_asr')
    timit_train_df = timit["train"].to_pandas()
    timit_test_df = timit["test"].to_pandas()
    return timit_train_df, timit_test_df


def select_different_speaker_idx(speakers, current_speaker, num_mix): 
    """
    Parameters
    ----------
    speakers: list
        A list of every unique speakers.
    current_speaker: str
        Current speaker in the speakers list.
    num_mix: int
        Number of speakers which is different from current speaker to choose.
    """
    # np.random.seed(914)
    idx = []
    for i in range(len(speakers)):
        if speakers[i] != current_speaker:
            idx.append(i)
    select_idx = list(np.random.choice(idx, num_mix, replace=False))
    return select_idx


def repeat_padding(src, trg):
    """
    Parameters
    ----------
    src: list or np.array
        Source array.
    trg: list or np.array
        Target array.
    
    References
    ----------
    1. https://stackoverflow.com/questions/60972141/how-to-perform-repeat-padding-in-numpy
    """
    data = [src, trg]
    m = len(max(data, key=len))
    r = np.array(list(starmap(np.resize, ((e, m) for e in data))))
    return r[0], r[1]


def get_speaker_mixed_waveforms(df, sample_rate, num_mix, snr):
    """
    Parameters
    ----------
    df: pd.DataFrame
        A dataframe which can be generated from 'get_timit_metadata()'.
    sample_rate: int
        Target sampling rate.
    num_mix: int
        Number of speakers which is different from current speaker to choose.
    snr: int
        Ratio of signal power to the noise power, often expressed in decibels.
    """
    speaker_mixed_waveforms = []
    for i in tqdm(range(len(df))):
        # Randomly select speaker which is different from current speaker
        current_speaker = df.speaker_id[i]
        select_indices = select_different_speaker_idx(
            speakers, 
            current_speaker=current_speaker, 
            num_mix=num_mix
        )

        for select_idx in select_indices:
            # Load two speakers' waveforms
            another_speaker = idx2spk[select_idx]
            speaker1_file_path = df.file[i]
            speaker2_file_path = df.query(f"speaker_id=='{another_speaker}'").file.values[0]
            speaker1_wav, _ = librosa.load(speaker1_file_path, sr=sample_rate)
            speaker2_wav, _ = librosa.load(speaker2_file_path, sr=sample_rate)

            # Calculate the scale to mix two speakers based on fixed SNR
            speaker2_power = np.mean(np.square(speaker1_wav)) / (10**(snr/10))
            scale = np.sqrt(speaker2_power / np.mean(np.square(speaker2_wav)))

            # Mix two speakers
            speaker1_length, speaker2_length = len(speaker1_wav), len(speaker2_wav)
            if speaker1_length == speaker2_length:
                speakers_mix = speaker1_wav + scale * speaker2_wav
            elif speaker1_length > speaker2_length:
                speaker1_wav, speaker2_wav_aug = repeat_padding(speaker1_wav, speaker2_wav)
                speakers_mix = speaker1_wav + scale * speaker2_wav_aug
            elif speaker1_length < speaker2_length:
                speakers_mix = speaker1_wav + scale * speaker2_wav[:len(speaker1_wav)]

            speaker_mixed_waveforms.append((
                current_speaker, 
                another_speaker, 
                speaker1_file_path, 
                speaker2_file_path, 
                speakers_mix
            ))
    
    return speaker_mixed_waveforms
