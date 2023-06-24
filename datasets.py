import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer


class TextDataset(Dataset):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    start_token_id = tokenizer.convert_tokens_to_ids(['<s>'])[0]
    end_token_id = tokenizer.convert_tokens_to_ids(['</s>'])[0]
    pad_token_id = tokenizer.convert_tokens_to_ids(['<pad>'])[0]

    def __init__(self, file_path):
        with open(file_path) as file:
            texts = [line.strip() for line in file]

        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        targets = self.tokenizer.convert_tokens_to_ids(text.split())
        inputs = [self.start_token_id] + targets[:-1]

        return inputs, targets

    @staticmethod
    def pad_2d(array_2d, pad_value=0):
        lengths = [len(row) for row in array_2d]
        max_length = max(lengths)
        for i in range(len(array_2d)):
            array_2d[i] += [pad_value for _ in range(max_length - lengths[i])]

        return array_2d

    @staticmethod
    def collate_fn(batch):
        inputs = []
        targets = []
        for cur_inputs, cur_targets in batch:
            inputs.append(cur_inputs)
            targets.append(cur_targets)

        inputs = TextDataset.pad_2d(inputs, pad_value=TextDataset.pad_token_id)
        targets = TextDataset.pad_2d(targets, pad_value=TextDataset.pad_token_id)

        inputs = torch.tensor(inputs)
        targets = torch.tensor(targets)

        return inputs, targets
