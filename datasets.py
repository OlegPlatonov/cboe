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
    def get_attention_mask(lengths):
        batch_size = len(lengths)
        max_length = max(lengths)
        attn_mask = torch.zeros(batch_size, max_length, max_length)
        for i, length in enumerate(lengths):
            attn_mask[i] = torch.eye(max_length)
            attn_mask[i, :length, :length] = torch.tril(torch.ones(length, length))

        attn_mask = 1 - attn_mask
        attn_mask[attn_mask == 1] = - torch.inf
        attn_mask = attn_mask.unsqueeze(1)

        return attn_mask

    @staticmethod
    def collate_fn(batch):
        inputs = []
        targets = []
        for cur_inputs, cur_targets in batch:
            inputs.append(cur_inputs)
            targets.append(cur_targets)

        lengths = [len(cur_inputs) for cur_inputs in inputs]

        inputs = TextDataset.pad_2d(inputs, pad_value=TextDataset.pad_token_id)
        targets = TextDataset.pad_2d(targets, pad_value=TextDataset.pad_token_id)

        inputs = torch.tensor(inputs)
        targets = torch.tensor(targets)

        attn_mask = TextDataset.get_attention_mask(lengths)

        return inputs, attn_mask, targets
