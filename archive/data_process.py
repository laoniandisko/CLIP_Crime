import os
from tqdm import tqdm


train_pathes = []
test_pathes = []
classes = []
for classe in tqdm(os.listdir("Train")):
    classes.append(classe)
    class_path = "Train/"+classe
    for path in os.listdir(class_path):
        true_path = "../"+"archive/"+class_path+"/"+path
        train_pathes.append(true_path+"\t"+classe)


for classe in tqdm(os.listdir("Test")):
    classes.append(classe)
    class_path = "Test/"+classe
    for path in os.listdir(class_path):
        true_path = "../"+"archive/"+class_path+"/"+path
        test_pathes.append(true_path+"\t"+classe)


with open("path_tag_train.txt","w",) as file:
    for i in train_pathes:
        file.write(i+"\n")

with open("path_tag_test.txt","w",) as file:
    for i in test_pathes:
        file.write(i+"\n")
