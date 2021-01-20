import pickle
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from attractor import BertAttractor
from torch.utils.data import DataLoader, TensorDataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

def load_tokens(token_ids_file, att_masks_file, labels_file):
    with open(token_ids_file, 'rb') as f:
        token_ids = pickle.load(f)

    with open(att_masks_file, 'rb') as f:
        att_masks = pickle.load(f)
    
    with open(labels_file, 'rb') as f:
        labels = pickle.load(f)
    
    return torch.tensor(token_ids), torch.tensor(att_masks), labels

def test(model, test_loader, device):
    model.eval()
    total_loss = total_acc = 0
    out_cls = None

    for batch_id, (token_id, att_mask) in enumerate(test_loader):
        with torch.no_grad():
            token_id, att_mask = token_id.to(device), att_mask.to(device)
            
            cls_embeds = model.cls_embeddings(token_id, att_mask).cpu().numpy()
            
            torch.cuda.empty_cache()
            
            out_cls = np.vstack((out_cls, cls_embeds)) if out_cls is not None else cls_embeds
    
    return out_cls

def evaluate(args):
    '''data loaders'''
    test_token_ids, test_att_masks, labels = load_tokens(args.test_sent_tokens, args.test_att_tokens, args.test_labels)
    test_dataset = TensorDataset(test_token_ids, test_att_masks)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    '''model setup'''
    model = BertAttractor()

    '''device allocation'''
    if torch.cuda.is_available():
        print('Using gpu')
        device = torch.device('cuda:1')
    else:
        print('Cannot evaluate without GPU')
        sys.exit()

    model.to(device)
    '''load the model'''
    checkpoint = torch.load(args.pt_model)
    model.load_state_dict(checkpoint['model_state_dict'])

    '''generate embeddings from the model'''
    out_cls = test(model, test_loader, device)
    
    '''tsne plot'''
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=400)
    tsne_results = tsne.fit_transform(out_cls)

    sent_dict = {0: "negative", 1:"neutral", 2:"positive"}
    named_labels = [sent_dict[x] for x in labels]

    print('tsne-computation done')
    plt.figure(figsize=(10,10))
    sns.scatterplot(tsne_results[:,0], tsne_results[:, 1], hue=named_labels, palette=['red', 'blue', 'green'])
    plt.savefig(args.out_dir + '/' + args.name + '.png', dpi=480)
    plt.close()
   
def main():
    parser = argparse.ArgumentParser(description='Evaluate the model labels')
    parser.add_argument('test_sent_tokens', help='Padded sentence tokens for testing')
    parser.add_argument('test_att_tokens', help='Padded attention tokens for testing')
    parser.add_argument('test_labels', help='test labels')
    parser.add_argument('out_dir', help='directory to store snapshots')
    parser.add_argument('pt_model', default=None, help='snapshot file')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--name', default='test', type=str, help='to save labels under what name')
    parser.add_argument('--max_len', default=64, type=int, help='maximum length of sentence in tokens')
    args = parser.parse_args()
    print(args)

    '''start training'''
    evaluate(args)

    print('Finished!')

if __name__=='__main__':
    main()
