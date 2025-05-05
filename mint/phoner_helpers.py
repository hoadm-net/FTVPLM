from os import path
import json
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from mint.config import DATA_DIR


class PhoNERDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class PhoNERDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        model_name="vinai/phobert-base-v2", 
        batch_size=16, 
        max_length=256
    ):
        super().__init__()

        PhoNER_DIR = path.join(DATA_DIR, "PhoNER_COVID19", "word")
        self.train_path = path.join(PhoNER_DIR, "train_word.json")
        self.val_path = path.join(PhoNER_DIR, "dev_word.json")
        self.test_path = path.join(PhoNER_DIR, "test_word.json")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.batch_size = batch_size
        self.max_length = max_length
        self.label2id = {}
        self.id2label = {}

    def prepare_data(self):
        # Read the data files
        def read_json(path):
            with open(path, 'r', encoding='utf-8') as f:
                return [json.loads(line) for line in f]
        
        self.train_data = read_json(self.train_path)
        self.val_data = read_json(self.val_path)
        self.test_data = read_json(self.test_path)

        # label2id from all datasets
        all_tags = set()
        for dataset in [self.train_data, self.val_data, self.test_data]:
            for item in dataset:
                all_tags.update(item['tags'])
        self.label2id = {tag: idx for idx, tag in enumerate(sorted(all_tags))}
        self.id2label = {v: k for k, v in self.label2id.items()}

        # Convert tags to ids
        def convert_tags(data):
            return [
                {
                    'words': item['words'],
                    'tags': [self.label2id[tag] for tag in item['tags']]
                }
                for item in data
            ]
        self.train_data = convert_tags(self.train_data)
        self.val_data = convert_tags(self.val_data)
        self.test_data = convert_tags(self.test_data)

    def collate_fn(self, batch):
        words = [item['words'] for item in batch]
        tags = [item['tags'] for item in batch]
        
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )
        
        labels = []
        for i, label in enumerate(tags):
            word_ids = encoding.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        
        encoding['labels'] = torch.tensor(labels)
        return encoding

    def train_dataloader(self):
        return DataLoader(
            PhoNERDataset(self.train_data),
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            PhoNERDataset(self.val_data),
            batch_size=self.batch_size,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            PhoNERDataset(self.test_data),
            batch_size=self.batch_size,
            collate_fn=self.collate_fn
        )
