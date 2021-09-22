import glob
import librosa
import torch
import os
import torchaudio
import opensmile
from tmh.audio_embeddings import get_audio_embeddings
# get embeddings

# path to data

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
)

path = "data/Second_DiCOVA_Challenge_Dev_Data_Release/AUDIO"
folders = ["breathing", "cough", "speech"]

def feature_extractor(path):

    for folder in folders:
        full_path = path + "/" + folder

        audio_files = glob.glob(full_path + "/*.16k.flac")
        print(audio_files)

        for audio_file in audio_files:
            print("the audio file is", audio_files)

            ## mel spectogram mfcc

            ## only do if no mel exist
            # if not os.path.isfile(audio_file + "mel_spectogram.pt"):
            
            # get all the files from the cuts folder

    
            waveform, sample_rate = torchaudio.load(audio_file)
            # resampled_file = change_sample_rate(audio_path=audio_file, new_sample_rate=16000)
            # sample_rate=16000

            # mel + mfcc

            mel_spectogram = torchaudio.transforms.MelSpectrogram()(waveform)
            mfcc = torchaudio.transforms.MFCC()(waveform)
            torch.save(mel_spectogram, audio_file + "mel_spectogram.pt")
            torch.save(mfcc, audio_file + "mfcc.pt")

            # egemaps
            ege = smile.process_signal(waveform,sample_rate)
            print("the ege file is", ege)
            ege.to_csv(audio_file + "egemaps.csv")

            # wav2vec2 embeddings
            audio_embeddings = get_audio_embeddings(audio_file)
            print(audio_embeddings)
            torch.save(audio_embeddings, audio_file + "hubert.pt")



def change_sample_rate(audio_path, new_sample_rate=16000):
    audio_to_resample, sr = librosa.load(audio_path)
    resampled_audio = librosa.resample(audio_to_resample, sr, new_sample_rate)
    resampled_tensor = torch.tensor([resampled_audio])
    return resampled_tensor


def create_spectogram_and_mfcc_from_path(path):
    audio_files = glob.glob(path + "/*.wav")
    print(audio_files)

    for audio_file in audio_files:
        waveform, sample_rate = torchaudio.load(audio_file)

        mel_spectogram = torchaudio.transforms.MelSpectrogram()(waveform)
        mfcc = torchaudio.transforms.MFCC()(waveform)

        ## save to path
        torch.save(mel_spectogram, audio_file + "mel_spectogram.pt")
        torch.save(mfcc, audio_file + "mfcc.pt")

    
feature_extractor(path)