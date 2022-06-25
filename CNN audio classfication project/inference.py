import torch
import torchaudio
from cnnModel import CNNNetwork
from rehapeSampleDatasets import UrbanSoundDataset
from train import AUDIO_DIR, ANNOTATIONS_FILE, SAMPLE_RATE, NUM_SAMPLES

total = 8732
index, miss = 0, 0

class_mapping = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music"
]


def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected


if __name__ == "__main__":
    # load back the model
    cnn = CNNNetwork()
    state_dict = torch.load("cnnnet.pth")
    cnn.load_state_dict(state_dict)

    # load urban sound dataset dataset
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    used = UrbanSoundDataset(ANNOTATIONS_FILE,
                             AUDIO_DIR,
                             mel_spectrogram,
                             SAMPLE_RATE,
                             NUM_SAMPLES,
                             "cpu")

    for index in range(total):
        # get a sample from the urban sound dataset for inference
        input, target = used[index][0], used[index][1]  # [batch size, num_channels, fr, time]
        input.unsqueeze_(1)
        index = index + 1
        # make an inference
        predicted, expected = predict(cnn, input, target,
                                      class_mapping)
        if predicted != expected:
            miss = miss + 1
        # print(f"Predicted: '{predicted}', expected: '{expected}'")

    # Accuracy calculate
    hit = total - miss
    Accuracy = hit / total * 100
    print(f"Accuracy: '{round(Accuracy, 2)}'")
