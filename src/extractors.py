from featureExtractor import FeatureExtractor
import numpy as np

# Make sure to write doc string to describe the functon for logging

def extract_mfcc_mean_std_26(audio, sr):
    """
    Extracts MFCC features from the given audio and computes their mean and standard deviation.
    Returns a concatenated feature vector of size (26 * 2).
    """
    extractor = FeatureExtractor(sr)
    mfccs = extractor.extract_mfcc(audio, 26)
    
    mfcc_mean = np.mean(mfccs, axis = 1)
    mfcc_std = np.std(mfccs, axis = 1)

    return np.concatenate([mfcc_mean, mfcc_std])
