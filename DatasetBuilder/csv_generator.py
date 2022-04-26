import os
import pathlib 
import pandas as pd


PATH = "C:/Projects/TFM/dataset/AD_MCI_HC_WINDOWED"
SAVE_PATH = "C:/Projects/TFM/dataset/AD_MCI_HC_WINDOWED/data.csv"

def main(root_path, save_path):
    elements = []
    root_path = pathlib.Path(root_path).resolve()
    class_folders = [ root_path.joinpath(folder) for folder in os.listdir(root_path) ]
    
    for subfolder in class_folders:
        for record in os.listdir(subfolder):
            if not ".mat" in record: continue
            elements.append({
                "path": str(subfolder.joinpath(record)),
                "label": subfolder.stem
            })
            
    df = pd.DataFrame(elements)
    df.to_csv(SAVE_PATH)


if __name__ == "__main__":
    #main(PATH, SAVE_PATH)
    df = pd.read_csv(SAVE_PATH, index_col="Unnamed: 0")
    print(df.head())