import os 
import pathlib
from matplotlib.pyplot import stem
import scipy.io as sio
from data_reader import read_record



ROOT_PATH = "C:/Projects/TFM/dataset/AD_MCI_HC"
NEW_DIR = "C:/Projects/TFM/dataset/AD_MCI_HC_WINDOWED"
WINDOW_DURATION = 5
SAMPLING_FREQUENCY = 256
N = WINDOW_DURATION * SAMPLING_FREQUENCY


def save_mat(file_name, matrix, current_class, idx):
    save_dict = {"EEG": matrix}
    name  = file_name.stem 
    new_path = pathlib.Path(NEW_DIR).resolve().joinpath(current_class, f"{name}_{idx}.mat")
    print(new_path)
    sio.savemat(new_path, save_dict)

def window(file_name, data, current_class):
    max_size = data.shape[1]
    idx = 0
    for index in range(0, max_size, N):
        chunk = data[:, index:(index+N)]
        if chunk.shape[1] < N:
            break
        save_mat(file_name, chunk, current_class, idx)
        idx += 1
        
    

def main():
    root_path = pathlib.Path(ROOT_PATH).resolve()
    subfolders = [ root_path.joinpath(folder) for folder in os.listdir(root_path) ]
    
    for folder in subfolders:
        
        class_records = [ folder.joinpath(record) for record in os.listdir(folder) if ".mat" in record ]
        
        for record in class_records:
            try:
                current_class = folder.stem
                print(current_class)
                data = read_record(record)
                window(record, data, current_class)
                print(f"{record}: {data.shape}")
            except Exception:
                print(f"Error opening file {record}")

if __name__ == "__main__":
    main()