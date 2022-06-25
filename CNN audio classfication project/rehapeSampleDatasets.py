import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio


class UrbanSoundDataset(Dataset):

    def __init__(self,
                 annotationFile,
                 audio_dir,
                 transformation,
                 targetSampleRate,
                 numSamples,
                 device):
        self.annotations = pd.read_csv(annotationFile)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.targetSampleRate = targetSampleRate
        self.numSamples = numSamples

    def __len__(self):
        return len(self.annotations)

    # len(used)

    def __getitem__(self, index):
        audio_sample_path = self.get_audio_sample_path(index)
        label = self.get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device) # to cuda
        # signal ->(num_channels, samples) -> (2,16000) -> (1, 16000)
        signal = self.resample(signal, sr)
        signal = self.mix_down(signal)
        signal = self.cut(signal)
        signal = self.right_pad(signal)
        signal = self.transformation(signal)
        return signal, label

    # a_list[1] => a_list.__getitem__(1)

    # ================================================================================================
    # data reshape

    def resample(self, signal, sr):
        if sr != self.targetSampleRate:
            resampler = torchaudio.transforms.Resample(sr, self.targetSampleRate).to(self.device)
            signal = resampler(signal)
        return signal

    def mix_down(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def cut(self, signal):
        # signal - > Tensors -> (1, num_samples) -> (1, 22050)
        if signal.shape[1] > self.numSamples:
            signal = signal[:, :self.numSamples]
        return signal

    def right_pad(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.numSamples:
            num_missing_samples = self.numSamples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    # ====================================================================================================

    def get_audio_sample_path(self, index):
        fold = f"fold{self.annotations.iloc[index, 5]}"
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[index, 0])
        return path

    def get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 6]


if __name__ == "__main__":
    ANNOTATIONS_FILE = "D:/Python/Project/ECE616/Project8K/data/UrbanSound8K/metadata/UrbanSound8K.csv"
    AUDIO_DIR = "D:/Python/Project/ECE616/Project8K/data/UrbanSound8K/audio"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device} device")

    # define transformation
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
    # ms = mel_spectrogram(signal)

    used = UrbanSoundDataset(ANNOTATIONS_FILE,
                             AUDIO_DIR,
                             mel_spectrogram,
                             SAMPLE_RATE,
                             NUM_SAMPLES,
                             device)

    print(f"There are {len(used)} samples in the dataset.")

    # file extract test↓
    # signal, label = used[0]
