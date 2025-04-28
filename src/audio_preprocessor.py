import numpy as np
import librosa
import noisereduce as nr

class AudioPreprocessor:
    def __init__(self, target_sample_rate=16000):
        self.target_sample_rate = target_sample_rate

    def load_audio(self, file_path: str):
        audio, sr = librosa.load(file_path, sr=self.target_sample_rate)
        return audio, sr

    def reduce_noise(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Applies noise reduction to the input audio waveform.
        """
        reduced_audio = nr.reduce_noise(y=audio, sr=sample_rate)
        return reduced_audio

    def remove_silence(self, audio: np.ndarray, top_db: int = 20) -> np.ndarray:
        """
        Remove silent periods from an audio waveform.
        """
        non_silent_intervals = librosa.effects.split(y=audio, top_db=top_db)
        non_silent_audio = np.concatenate([audio[start:end] for start, end in non_silent_intervals])
        return non_silent_audio

    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize the audio to have consistent volume.
        """
        rms = np.sqrt(np.mean(audio**2))
        return audio / (rms + 1e-6)

    def preprocess(self, file_path: str):
        """
        Full preprocessing pipeline: load, remove silence, denoise, normalize.
        """
        audio, sr = self.load_audio(file_path)
        audio = self.remove_silence(audio)
        audio = self.reduce_noise(audio, sr)
        audio = self.normalize_audio(audio)
        return audio, sr
