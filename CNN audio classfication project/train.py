import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
from rehapeSampleDatasets import UrbanSoundDataset
from cnnModel import CNNNetwork

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.0001

ANNOTATIONS_FILE = "D:/Python/Project/ECE616/Project8K/data/UrbanSound8K/metadata/UrbanSound8K.csv"
AUDIO_DIR = "D:/Python/Project/ECE616/Project8K/data/UrbanSound8K/audio"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050


def set_data_loader(train_data, batch_size):
    trainDataloader = DataLoader(train_data, batch_size=batch_size)
    return trainDataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)

        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i + 1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print("---------------------------")
    print("Finished training")


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    # instantiating our dataset object and create data loader
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64)

    usd = UrbanSoundDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)

    train_dataloader = set_data_loader(usd, BATCH_SIZE)

    # construct model and assign it to device
    cnn = CNNNetwork().to(device)
    print(cnn)

    # initialise loss funtion + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(cnn, train_dataloader, loss_fn, optimiser, device, EPOCHS)

    # save model
    torch.save(cnn.state_dict(), "cnnnet.pth")
    print("Trained feed forward net saved at cnnnet.pth")