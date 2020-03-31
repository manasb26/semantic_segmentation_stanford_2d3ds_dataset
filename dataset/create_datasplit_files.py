import os

import argparse
parser = argparse.ArgumentParser(description='Optional app description')

parser.add_argument('--image_dir',
                    help='Reads image names to create train or test txt files.')

parser.add_argument('--file_name',
                    help='Dataset split filename: train.txt or test.')

parser.add_argument('--file_save_path',
                    help='Path of the directory to save file.')                    

args = parser.parse_args()

IMAGE_DIR_PATH = args.image_dir
DATASET_SPLIT_FILENAME = args.file_name
FILE_SAVE_PATH = args.file_save_path

fileNames = []

# r=root, d=directories, f = filesls
for r, d, f in os.walk(IMAGE_DIR_PATH):
    for file in f:
        fileNames.append(file.split(".")[0])

filePath = os.path.join(FILE_SAVE_PATH, DATASET_SPLIT_FILENAME+".txt")
createFile = open(filePath, "a+")

for fileName in fileNames:
    fileName = fileName[:-4]
    createFile.write("%s\n"%fileName)

print(len(fileNames))

createFile.close()