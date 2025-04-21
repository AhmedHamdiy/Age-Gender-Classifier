import librosa
import librosa.display
import numpy as np
import math
import matplotlib.pyplot as plt


class FeatureExtractor:
    def __init__(self, sr=22050, frame_size=2048, hop_length=512):
        self.sr = sr
        self.frame_size = frame_size
        self.hop_length = hop_length

    def extract_features(self, audio):
        features = {}

        # Time domain features
        features["zcr"] = self.extract_zcr(audio)
        features["rms"] = self.extract_rms(audio)
        features["amplitude_envelope"] = self.extract_amplitude_envelope(audio)

        # Frequency domain features
        features["spectral_centroid"] = self.extract_spectral_centroid(audio)
        features["spectral_bandwidth"] = self.extract_spectral_bandwidth(audio)
        features["band_energy_ratio"] = self.band_energy_ratio(
            audio, split_frequency=1000
        )
        features["spectral_rolloff"] = self.extract_spectral_rolloff(audio)
        features["spectral_contrast"] = self.extract_spectral_contrast(audio)
        features["spectral_flatness"] = self.extract_spectral_flatness(audio)
        features["mfcc"] = self.extract_mfcc(audio)
        features["mel_spectrogram"] = self.mel_spectrogram(audio)
        features["spectrogram"] = self.extract_spectrogram(audio)
        return features

    def extract_zcr(self, audio):
        return librosa.feature.zero_crossing_rate(
            audio, frame_length=self.frame_size, hop_length=self.hop_length
        )[0]

    def extract_rms(self, audio):
        return librosa.feature.rms(
            y=audio, frame_length=self.frame_size, hop_length=self.hop_length
        )[0]

    def extract_amplitude_envelope(self, audio):
        return np.array(
            [
                max(audio[i : i + self.frame_size])
                for i in range(0, len(audio), self.hop_length)
            ]
        )

    def extract_spectral_centroid(self, audio):
        return librosa.feature.spectral_centroid(
            y=audio, sr=self.sr, n_fft=self.frame_size, hop_length=self.hop_length
        )[0]

    def extract_spectral_bandwidth(self, audio):
        return librosa.feature.spectral_bandwidth(
            y=audio, sr=self.sr, n_fft=self.frame_size, hop_length=self.hop_length
        )[0]

    def band_energy_ratio(self, audio, split_frequency=1000):
        spectrogram = librosa.stft(
            audio, n_fft=self.frame_size, hop_length=self.hop_length
        )

        num_frequency_bins = self.frame_size // 2 + 1
        frequency_range = self.sr / 2
        frequency_delta_per_bin = frequency_range / num_frequency_bins
        split_frequency_bin = int(math.floor(split_frequency / frequency_delta_per_bin))

        power_spectrogram = np.abs(spectrogram) ** 2
        power_spectrogram = power_spectrogram.T

        band_energy_ratio = []
        for frame in power_spectrogram:
            sum_power_low = frame[:split_frequency_bin].sum()
            sum_power_high = frame[split_frequency_bin:].sum()
            ber = sum_power_low / (sum_power_high + 1e-10)
            band_energy_ratio.append(ber)

        return np.array(band_energy_ratio)

    def extract_spectral_rolloff(self, audio):
        return librosa.feature.spectral_rolloff(
            y=audio, sr=self.sr, n_fft=self.frame_size, hop_length=self.hop_length
        )[0]

    def extract_spectral_contrast(self, audio):
        contrast = librosa.feature.spectral_contrast(
            y=audio, sr=self.sr, n_fft=self.frame_size, hop_length=self.hop_length
        )
        return np.mean(contrast, axis=0)

    def extract_spectral_flatness(self, audio):
        return librosa.feature.spectral_flatness(
            y=audio, n_fft=self.frame_size, hop_length=self.hop_length
        )[0]
    
    def extract_spectrogram(self, audio):
        return librosa.amplitude_to_db(librosa.stft(audio), ref=np.max)
    
    def mel_spectrogram(self,audio):
        return librosa.feature.melspectrogram(
            y=audio, sr=self.sr, hop_length=self.hop_length, n_mels=128
        )[0]


    def extract_mfcc(self, audio, n_mfcc=13):
        mfccs = librosa.feature.mfcc(
            y=audio, sr=self.sr, n_fft=self.frame_size, hop_length=self.hop_length, n_mfcc=n_mfcc
        )
        return np.mean(mfccs, axis=0)

    def plot_features(self, audio, features=None):
        if features is None:
            features = self.extract_features(audio)

        plot_configs = [
            {"feature": "zcr", "color": "r", "title": "Zero Crossing Rate"},
            {"feature": "rms", "color": "g", "title": "RMS Energy"},
            {"feature": "amplitude_envelope", "color": "b", "title": "Amplitude Envelope"},
            {"feature": "spectral_centroid", "color": "m", "title": "Spectral Centroid"},
            {"feature": "spectral_bandwidth", "color": "c", "title": "Spectral Bandwidth"},
            {"feature": "band_energy_ratio", "color": "y", "title": "Band Energy Ratio"},
            {"feature": "spectral_rolloff", "color": "pink", "title": "Spectral Rolloff"},
            {"feature": "spectral_contrast", "color": "orange", "title": "Spectral Contrast"},
            {"feature": "spectral_flatness", "color": "purple", "title": "Spectral Flatness"},
            {"feature": "mfcc", "color": "brown", "title": "MFCC"},
            {"feature": "mel_spectrogram", "color": "red", "title": "Mel Spectrogram"},
            {"feature": "spectrogram", "color": "blue", "title": "Spectrogram"}
        ]

        num_features = len(plot_configs)
        num_rows = (num_features + 1) // 2  # Ceiling division
        plt.figure(figsize=(14, 4 * num_rows))

        for i, config in enumerate(plot_configs, 1):
            feature_name = config["feature"]
            feature_data = features[feature_name]

            # Handle 2D features (like those from librosa)
            if isinstance(feature_data, np.ndarray) and feature_data.ndim > 1:
                feature_data = feature_data[0]  # Take first (and only) row

            plt.subplot(num_rows, 2, i)
            frames = range(len(feature_data))
            t = librosa.frames_to_time(frames, hop_length=self.hop_length)

            librosa.display.waveshow(audio, sr=self.sr, alpha=0.5)
            plt.plot(t, feature_data, color=config["color"], label=config["title"])
            plt.title(f'Waveform with {config["title"]}')
            plt.legend()

        plt.tight_layout()
        plt.show()
