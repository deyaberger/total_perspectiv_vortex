import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score

from mne import Epochs, pick_types, events_from_annotations
import mne.channels as channels
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP
from mne import set_log_level
set_log_level("WARNING")


class Visualizer():
    """Class to Visualize MNE data."""

    def __init__(self):
        pass

    @classmethod
    def show_montage(self, montage):
        """Plot EEG montage."""
        montage.plot()
        plt.show()

    @classmethod
    def show_raw_data(self, raw, noiiice=False):
        """Plot row data (Volts of signals for each channel along the time)."""
        print("Plotting raw data\n")
        if noiiice:
            raw.plot(scalings=dict(eeg=1e-4), title="Raw Data")
        else:
            raw.plot(title="Raw Data")
        plt.show()

    @classmethod
    def show_psd(self, raw, average=False):
        """Plot power spectral density (PSD)."""
        if average:
            raw.compute_psd().plot(average=True, color='blue')
        else:
            raw.compute_psd().plot()
        plt.show()


class Parser():
    """Class to Parse MNE data."""

    def __init__(self, **kwargs):
        """Initialize data path, subject number and run task."""
        self.subject = kwargs['subject']
        self.runs = kwargs['runs']
        self.mne_path = kwargs['mne_path']

    def load_data(self):
        """Load data."""
        print(f"Loading data for subject {self.subject}, at runs {self.runs}\n")
        raw_fnames = eegbci.load_data(self.subject, self.runs, path=self.mne_path)
        self.raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
        eegbci.standardize(self.raw)

    def select_montage(self, montage_name):
        """Select the montage that better fits motor imagery task."""
        print(f"Selecting montage: {montage_name}\n")
        montage = make_standard_montage(montage_name)
        self.raw.set_montage(montage, on_missing='ignore')
        return montage

    def compute_psd(self):
        """Compute power spectral density (PSD)."""
        self.raw.compute_psd()

    def reduce_noise(self, noisy_freq, noisy_channels):
        """Reducing noise to better analyse the data.

        - noisy_freq = Power line noise is a type of interference that can be picked up by EEG electrodes due to
                        the electrical activity of nearby power lines
        - noisy_channels = Channels that do are not related to the motion cortex.
        """
        print("Reducing noise\n")
        self.raw.notch_filter(noisy_freq, method="iir")
        picks = pick_types(self.raw.info, eeg=True,
                           exclude=noisy_channels)
        self.raw.pick(picks)

    def focus_and_clean(self, significant_frequencies):
        """Focus on the frequencies that are relevant for motor imagery."""
        print("Focussing on significant frequencies\n")
        start, end = significant_frequencies
        self.raw.filter(start, end)


if __name__ == '__main__':
    show_plots = True
    parser = Parser(subject=1, runs=[3, 7, 11], mne_path="./mne_data")
    parser.load_data()
    montage = parser.select_montage('biosemi64')
    if show_plots:
        Visualizer.show_montage(montage)
        Visualizer.show_raw_data(parser.raw)
        Visualizer.show_raw_data(parser.raw, noiiice=True)
        Visualizer.show_psd(parser.raw)
    if not show_plots:
        parser.compute_psd()
    noisy_channels = ['AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'Fp1', 'Fpz', 'Fp2', 'P7', 'P5', 'P3', 'P1', 'P2', 'P4', 'P6',
                      'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'Iz']
    parser.reduce_noise(60, noisy_channels)
    if show_plots:
        Visualizer.show_psd(parser.raw)
    parser.focus_and_clean(significant_frequencies=(8.0, 40.0))
    if show_plots:
        Visualizer.show_psd(parser.raw)
        Visualizer.show_psd(parser.raw, average=True)
    print("Tadaaa")