from data_reader import read_record
#from data_explorer import records_per_class

if __name__ == "__main__":
    ROOT_PATH = r"C:\Projects\TFM\dataset\AD_MCI_HC"
    PATH = r"C:\Projects\TFM\dataset\AD_MCI_HC\HC\CNTR_25_T0_filtered_CLEAN.mat"
    record = read_record(PATH)
    print(record[1].shape)
"""    num_records = records_per_class(ROOT_PATH)
    print(num_records)
"""
