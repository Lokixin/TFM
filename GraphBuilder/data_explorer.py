import os
import pathlib
from statistics import mean, median, quantiles
import numpy as np
from matplotlib import pyplot as plt
from data_reader import read_record


class DataExplorer:
    """DataExplorer class provides a set of tools to explore
    the dataset. It can be used to retrieve the classes and its labels,
    the amount of files per class, the number of samples in the EEG recordings 
    and also to plot EEGs (full or individual channels).
    """
    def __init__(self, root_path: str) -> None:
        """Generate a DataExplorer object for a given directory
        storing a dataset.

        Args:
            root_path (str): Full path to the directory containing
            the dataset
        """
        self.root_path = pathlib.Path(root_path).resolve()
        self.class_folders = [ self.root_path.joinpath(folder) for folder in os.listdir(self.root_path) ]
        
    def records_per_class(self) -> dict:
        """ Finds all the different classes inside the dataset
        and the number of recordings for each one of them.

        Returns:
            records_amount (dict): {ClassName: numberOfRecords}
        """
        records_amount = { folder.stem: len(os.listdir(folder)) for folder in self.class_folders }
        return records_amount
    
    def min_max_sample_amount(self) -> tuple:
        """Find the minimum and the maximum amount of samples recorded 
        along all the files in the dataset.

        Returns:
            stats: A dict containing statistics about the 
            amount of samples found througout all the records
            inside the dataset.
        """
        samples_amount = []

        for subfolder in self.class_folders:
            for record in os.listdir(subfolder):
                if not ".mat" in record: continue
                try:
                    record_samples = read_record(subfolder.joinpath(record))
                    samples_amount.append(record_samples.shape[1])
                except Exception:
                    print(f"File: {subfolder.joinpath(record)} could not be readen")
        stats = {
            "min": min(samples_amount),
            "max": max(samples_amount),
            "mean": mean(samples_amount),
            "median": median(samples_amount),
            "std": np.std(samples_amount),
            "quantiles": quantiles(samples_amount)
        }
        return stats
    
    def plot_record(self, record: np.ndarray) -> None:
        """Plot all the channels in the same axis. The samples' index
        is its timestap.

        Args:
            record (np.ndarray): The EEG record matrix.
        """
        x_axis = self.__get_temporal_index(len(record[0]))
        for idx, channel in enumerate(record):
            plt.plot(x_axis, channel, label=f"Ch. {idx + 1}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.margins(x=0)
        plt.grid()
        plt.show()
            
    def plot_channel(self, record: np.ndarray, channel_idx: int) -> None:
        """Plots a single channel of a given record. 

        Args:
            record (np.ndarray): The EEG record matrix.
            channel_idx (int): Index of the channel to be ploted.
        """
        x_axis = self.__get_temporal_index(len(record[channel_idx]))
        plt.plot(x_axis, record[channel_idx], label=f"Ch. {channel_idx}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.margins(x=0)
        plt.grid()
        plt.show()
    
    def __get_temporal_index(self, num_samples: int, sample_rate: int =256) -> np.ndarray:
        """Converts the sample index to a temporal index for a given 
        sampling rate

        Args:
            num_samples (int): The number of samples of a channel
            sample_rate (int, optional): The sampling rate used to obtain 
            or downsample the recording. Defaults to 256.

        Returns:
            x_axis (np.ndarray): A temporal index
        """
        x_axis = np.arange(num_samples)
        x_axis = x_axis / sample_rate
        return x_axis





        