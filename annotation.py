
import os
from os import getcwd

dataset_path = 'dataset'
if __name__ == "__main__":
    wd = getcwd()
    
    list_file = open('dataset.txt', 'w')
    for filename in os.listdir(dataset_path):
        # print(os.path.join(wd,dataset_path, filename))
        path = os.path.join(wd,dataset_path, filename)
        list_file.write(path)
        list_file.write('\n')
    list_file.close()
    print("Generate Done!")
