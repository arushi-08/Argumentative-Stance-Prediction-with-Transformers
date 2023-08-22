import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class TextImageDescriptions(Dataset):
    def __init__(self, partition, image_prompts, dataset='abortion') -> None:
        super().__init__()
        self.dataset = dataset
        self.partition = partition
        self.tweets = pd.read_csv(f'../data/{dataset}_{partition}.csv')
        self.image_descriptions = pd.read_csv(f'../data/{dataset}_instructblip_descriptions.csv')
        self.image_descriptions['img'] = self.image_descriptions['img'].apply(lambda x: x.split('.')[0]).astype(int)
        self.dataset = pd.merge(self.image_descriptions, self.tweets, left_on='img', right_on='tweet_id')
        self.dataset = self.dataset[self.dataset['split'] == partition]
        self.promt_col = image_prompts
        self.label_assignments = {'yes': 1, 'no': 0}
        if isinstance(self.promt_col, list):
            self.promt_col = [i.replace(' ', '_') for i in self.promt_col]
            self.dataset['image_description'] = self.dataset[self.promt_col].agg('.'.join, axis=1)
        else:
            self.promt_col = self.promt_col.replace(' ', '_')
            self.dataset['image_description'] = self.dataset[self.promt_col]
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image_description = self.dataset.iloc[idx]['image_description']
        tweet_text = self.dataset.iloc[idx]['tweet_text']
        label = self.dataset.iloc[idx]['persuasiveness']
        label = self.label_assignments[label]
        # print(self.partition, idx, label)
        return image_description, tweet_text, label
    
    def class_weights(self):
        weights = 1/torch.Tensor(self.dataset['persuasiveness'].value_counts(normalize=True).values.tolist())
        print(f'Class weights: {weights}')
        return weights
    

def initialize_data(batchsize, promts):
    train_dataset = TextImageDescriptions('train', promts)
    # train_dataset.class_weights()
    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=False)
    valid_dataset = TextImageDescriptions('dev', promts)
    valid_loader = DataLoader(valid_dataset, batch_size=batchsize, shuffle=False)
    # test_dataset = TextImageDescriptions('test', promts)
    # test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True)
    test_loader = None
    return train_loader, valid_loader, test_loader
