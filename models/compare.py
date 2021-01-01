import pickle
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from attractor import SeqBert
from itertools import permutations
from torch.utils.data import DataLoader, TensorDataset


def load_tokens(token_ids_file, att_masks_file, labels_file):
    with open(token_ids_file, 'rb') as f:
        token_ids = pickle.load(f)

    with open(att_masks_file, 'rb') as f:
        att_masks = pickle.load(f)

    with open(labels_file, 'rb') as f:
        labels = pickle.load(f)

    return torch.tensor(token_ids), torch.tensor(att_masks), torch.tensor(labels).long()

def permutate(criterion, embed, label, device):
    e_perms = [torch.tensor(list(y)).view(1, -1).to(device) for y in permutations(embed)]
    all_loss = torch.stack([criterion(a_perm, label.view(1)) for a_perm in e_perms])
    return all_loss.min()

def batch_permutate(criterion, attractor_embeds, labels, device):
    total = 0
    for embed, label in zip(attractor_embeds, labels):
        total += permutate(criterion, embed, label, device)
    return total

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = total_acc = 0
    for batch_id, (token_id, att_mask, labels) in enumerate(train_loader):
        token_id, att_mask, labels = token_id.to(device), att_mask.to(device), labels.to(device)
        labels = labels.view(-1) 
    
        cls_embeds = model(token_id, att_mask)

        cls_loss = criterion(cls_embeds, labels)
        loss = cls_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.empty_cache()

        total_loss += loss.item()
        total_acc += (torch.argmax(cls_embeds, axis=1) == labels).sum()

        if (batch_id % 500 == 0) or (batch_id == len(train_loader) - 1):
            print(f'\t\tTrain iter {batch_id + 1 }/{len(train_loader)}')

    train_acc = total_acc.item() / len(train_loader.dataset)
    train_loss = total_loss / len(train_loader)

    return train_acc, train_loss

def test(model, test_loader, criterion, device):
    model.eval()
    total_loss = total_acc = 0
    for batch_id, (token_id, att_mask, labels) in enumerate(test_loader):
        with torch.no_grad():
            token_id, att_mask, labels = token_id.to(device), att_mask.to(device), labels.to(device)
            labels = labels.view(-1)
            
            cls_embeds = model(token_id, att_mask)

            cls_loss = criterion(cls_embeds, labels)
            loss = cls_loss

            torch.cuda.empty_cache()

            total_loss += loss.item()
            total_acc += (torch.argmax(cls_embeds, axis=1) == labels).sum()

            if (batch_id % 500 == 0) or (batch_id == len(test_loader) - 1):
                print(f'\t\tTest iter {batch_id + 1}/{len(test_loader)}')

    test_acc = total_acc.item() / len(test_loader.dataset)
    test_loss = total_loss / len(test_loader)

    return test_acc, test_loss            

def save_snapshot(epoch, model, optimizer, train_acc, train_loss, test_acc, test_loss, out_dir):
    torch.save({'epoch': epoch+1, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': train_acc, 'train_loss': train_loss,
                'test_acc': test_acc, 'test_loss': test_loss},
                out_dir + '/snapshot-' + str(epoch+1) + '.pt')

def train_model(model, train_loader, test_loader, optimizer, criterion, device,
                start_epoch, max_epochs, train_acc, train_loss, test_acc, test_loss, 
                out_dir, scheduler): 

    model.to(device)

    for epoch in range(start_epoch, max_epochs):
        print(f'Epoch: {epoch+1}/{max_epochs}')
        #train
        train_acc_epoch, train_loss_epoch = train(model, train_loader, criterion, optimizer, device)
        #test
        test_acc_epoch, test_loss_epoch = test(model, test_loader, criterion, device)

        train_acc.append(train_acc_epoch), train_loss.append(train_loss_epoch)
        test_acc.append(test_acc_epoch), test_loss.append(test_loss_epoch)

        #scheduler.step()
        print(f'\tTrain_acc: {train_acc_epoch}\tTrain_loss: {train_loss_epoch}')
        print(f'\tTest_acc: {test_acc_epoch}\tTest_loss: {test_loss_epoch}')

        save_snapshot(epoch, model, optimizer, train_acc, train_loss, test_acc, test_loss, out_dir)

    model.to('cpu')

def start_train(args):
    '''data loaders'''
    train_token_ids, train_att_masks, train_labels = load_tokens(args.train_sent_tokens, args.train_att_tokens, args.train_labels)
    train_dataset = TensorDataset(train_token_ids, train_att_masks, train_labels.reshape(-1,1))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    test_token_ids, test_att_masks, test_labels = load_tokens(args.test_sent_tokens, args.test_att_tokens, args.test_labels)
    test_dataset = TensorDataset(test_token_ids, test_att_masks, test_labels.reshape(-1,1))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    '''model setup'''
    model = SeqBert()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    '''device allocation'''
    if torch.cuda.is_available():
        print('Using gpu')
        device = torch.device('cuda:1')
    else:
        print('Cannot train without GPU')
        sys.exit()

    train_acc, train_loss = [], []
    test_acc, test_loss = [], []
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        train_acc, train_loss = checkpoint['train_acc'], checkpoint['train_loss']
        test_acc, test_loss = checkpoint['test_acc'], checkpoint['test_loss']
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)

    '''freeze bert model'''
#    for param in model.parameters():
#        param.requires_grad = False
#    for param in model.bert.classifier.parameters():
#        param.requires_grad = True
#    for param in model.attractor.parameters():
#        param.requires_grad = True

    '''train the model'''
    train_model(model, train_loader, test_loader, optimizer, criterion, device,
                start_epoch, args.epochs, train_acc, train_loss, test_acc, test_loss,
                args.out_dir, scheduler)

def main():
    parser = argparse.ArgumentParser(description='Phase 1 Training for the LSTMs')
    parser.add_argument('train_sent_tokens', help='Padded sentence tokens for training')
    parser.add_argument('train_att_tokens', help='Padded attention tokens for training')
    parser.add_argument('train_labels', help='train labels')
    parser.add_argument('test_sent_tokens', help='Padded sentence tokens for testing')
    parser.add_argument('test_att_tokens', help='Padded attention tokens for testing')
    parser.add_argument('test_labels', help='test labels')
    parser.add_argument('out_dir', help='directory to store snapshots')
    parser.add_argument('--epochs', default=2, type=int, help='maximum epochs to be performed')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--resume', default=None, help='snapshot file')
    parser.add_argument('--max_len', default=64, type=int, help='maximum length of sentence in tokens')
    args = parser.parse_args()
    print(args)

    '''start training'''
    start_train(args)

    print('Finished!')

if __name__=='__main__':
    main()
