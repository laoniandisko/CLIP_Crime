
import clip

from torch.utils.data import Dataset, DataLoader, ConcatDataset

class YourDataset(Dataset):
    def __init__(self,img_root,meta_root,is_train,preprocess):
        # 1.根目录(根据自己的情况更改)
        self.img_root = img_root
        self.meta_root = meta_root
        # 2.训练图片和测试图片地址(根据自己的情况更改)
        self.train_set_file = meta_root+'/path_tag_train.txt'
        self.test_set_file = meta_root+'/path_tag_test.txt'
        # 3.训练 or 测试(根据自己的情况更改)
        self.is_train = is_train
        # 4.处理图像
        self.img_process = preprocess
        # 5.获得数据(根据自己的情况更改)
        self.samples = []
        self.sam_labels = []
        # 5.1 训练还是测试数据集
        self.read_file = ""
        if is_train:
            self.read_file = self.train_set_file
        else:
            self.read_file = self.test_set_file
		# 5.2 获得所有的样本(根据自己的情况更改)
        with open(self.read_file,'r') as f:
            path_tag = f.readlines()
            for line in path_tag:
                img_path = line.split("\t")[0]
                label = line.split("\t")[1]
                label = "a photo of " + label
                self.samples.append(img_path)
                self.sam_labels.append(label)
        # 转换为token
        self.tokens = clip.tokenize(self.sam_labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        token = self.tokens[idx]
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        # 对图像进行转换
        image = self.img_process(image)
        return image,token