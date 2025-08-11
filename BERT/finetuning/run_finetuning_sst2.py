import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm.auto import tqdm

from run_pretraining import count_parameters
from transformer_layers import *
from bert_pretraining import *
from huggingface_tokenizer import HuggingFaceTokenizer

class SST2Dataset(Dataset):
    def __init__(self, tokenized_data):
        self.input_ids = tokenized_data['input_ids']
        self.token_type_ids = tokenized_data['token_type_ids']
        self.attention_mask = tokenized_data['attention_mask']
        self.labels = tokenized_data['label']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        input_ids = torch.tensor(self.input_ids[index], dtype=torch.long)
        token_type_ids = torch.tensor(self.token_type_ids[index], dtype=torch.long)
        attention_mask = torch.tensor(self.attention_mask[index], dtype=torch.long)
        labels = torch.tensor(self.labels[index], dtype=torch.long)
        return (input_ids, token_type_ids, attention_mask, labels)

def get_loader2(examples, batch_size=32, shuffle=None, pin_memory=True, num_workers=4):
    dataset = SST2Dataset(examples)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                        pin_memory=pin_memory, num_workers=num_workers)
    return loader

class BertBinaryClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = BertEmbedding(config)
        self.layers = nn.ModuleList([TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.num_labels = 2 # for binary classification
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

    def forward(self, input_ids, segment_ids, attn_mask):
        attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
        hidden_state = self.embedding(input_ids, segment_ids)

        for mod in self.layers:
            hidden_state = mod(hidden_state, attn_mask)

        cls_output = hidden_state[:, 0]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits

def train_finetune(model, train_loader, criterion, optimizer, scheduler, config):
    model.train()
    total_loss, total_acc = 0, 0

    for batch in tqdm(train_loader, desc='Training...'):
        input_ids, segment_ids, input_mask, labels = [item.to(config.device) for item in batch]

        optimizer.zero_grad()
        logits = model(input_ids, segment_ids, input_mask)

        loss = criterion(logits, labels) # torch.tensor
        preds = logits.argmax(dim=1)
        acc = (preds == labels).sum().item() / labels.size(0) # python float

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() # type(loss): torch.tensor
        total_acc += acc # type(acc): float
    return total_loss / len(train_loader), total_acc / len(train_loader)

def evaluate_finetune(model, eval_loader, criterion, config):
    model.eval()
    total_loss, total_acc = 0, 0

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc='Evaluating...'):
            input_ids, segment_ids, input_mask, labels = [item.to(config.device) for item in batch]

            logits = model(input_ids, segment_ids, input_mask)
            loss = criterion(logits, labels)

            preds = logits.argmax(dim=1)
            acc = (preds == labels).sum().item() / labels.size(0)

            total_loss += loss.item()
            total_acc += acc
    return total_loss / len(eval_loader), total_acc / len(eval_loader)


if __name__ == '__main__':

    config = PreTrainingBertConfig()
    tokenizer = HuggingFaceTokenizer(config.tokenizer_path)
    pretrained_model_path = 'pretraining_model.pt'

    sst2 = load_dataset('stanfordnlp/sst2')
    tokenizer = HuggingFaceTokenizer('bert-base-uncased').tokenizer

    def tokenization(examples):
        return tokenizer(examples['sentence'], truncation=True, padding='max_length', max_length=128)

    tokenized_sst2 = sst2.map(tokenization, batched=True, remove_columns=(['idx', 'sentence']))

    train_dataset = tokenized_sst2['train']
    split_dataset = train_dataset.train_test_split(test_size=0.2, seed=42)

    train_set = split_dataset['train']
    valid_set = split_dataset['test']
    test_set = tokenized_sst2['validation']

    train_loader = get_loader2(train_set, shuffle=True)
    valid_loader = get_loader2(valid_set, shuffle=False)
    test_loader = get_loader2(test_set, shuffle=False)

    model = BertBinaryClassifier(config).to(config.device)
    pretrained_dict = torch.load(pretrained_model_path, map_location=torch.device('cuda'))
    model.load_state_dict(pretrained_dict, strict=False)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    criterion = nn.BCEWithLogitsLoss()
    epochs = 1
    total_steps = len(train_loader) * epochs
    optimizer, scheduler = optimizer_and_scheduler(model, total_steps, config)

    best_loss = float('inf')
    for epcoh in range(epochs):
        print('Fine-tuning started...')
        train_loss, train_acc = train_finetune(model, train_loader, criterion, optimizer, scheduler, config)
        valid_loss, valid_acc = evaluate_finetune(model, valid_loader, criterion, config)
        print(f'Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f}')
        print(f'Valid Loss: {valid_loss:.4f} | Valid Accuracy: {valid_acc:.4f}')

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), 'sst2_finetuned_model.pt')
    print('Fine-tuning complete...')

    model.load_state_dict(torch.load('sst2_finetuned_model.pt'))
    test_loss, test_acc = evaluate_finetune(model, test_loader, criterion, config)
    print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}%')

