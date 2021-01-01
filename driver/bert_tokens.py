import transformers
import pandas as pd
import numpy as np
import argparse
from transformers import BertTokenizer
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.sequence import pad_sequences

def read_file(f_name):
    dataset = pd.read_csv(f_name)
    dataset.dropna(inplace=True)
    return dataset

def bert_tokens(tokenizer, dataset, max_len):
    token_ids = []
    lengths = []
    for row in dataset.index:
        sent = dataset.loc[row, 'text']
        encoded_sent = tokenizer.encode(sent, add_special_tokens=True)
        if len(encoded_sent) > max_len:
            encoded_sent = encoded_sent[:max_len]
            encoded_sent[max_len - 1] = 102
        token_ids.append(encoded_sent)
        lengths.append(len(encoded_sent))
    return token_ids, lengths

def bert_att_mask(sent_tokens):
    sent_att_masks = []
    for tokens in sent_tokens:
        att_mask = [int(token > 0) for token in tokens]
        sent_att_masks.append(att_mask)
    return np.array(sent_att_masks)

def padding(sent_tokens, max_len):
    padded_sent = pad_sequences(sent_tokens, maxlen=max_len, dtype="long", value=0, truncating="post", padding="post")
    return padded_sent

def validate_lengths(lengths):
    return np.any(np.array(lengths) > 512)

def save_tokens(sent_tokens, out_dir, name="train", pad=""):
    with open(out_dir + '/' + name + pad + '_tokens.pkl', 'wb') as f:
         pickle.dump(sent_tokens, f)

def load_tokens(out_dir, name="train", pad=""):
    with open(out_dir + '/' + name + pad + '_tokens.pkl', 'rb') as f:
        sent_tokens = pickle.load(f)
    return sent_tokens

def plot_dist_len(sent_tokens, out_dir, name="train"):
    lengths = [len(tokens) for tokens in sent_tokens]
    plt.figure(figsize=(16, 8))
    sns.countplot(x=lengths)
    plt.xlabel('token lengths')
    plt.savefig(out_dir + '/' + name + '_token_lens.png', dpi=480)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Bert Tokenizer Worker', add_help=True)
    parser.add_argument('train', help='Train csv')
    parser.add_argument('test', help='Test csv')
    parser.add_argument('out_dir', help='output directory')
    parser.add_argument('--stage', type=int, help='resume analysis stage')
    parser.add_argument('--case', default='uncased', choices=['uncased', 'cased'], help='which vocabulary to use')
    parser.add_argument('--which_bert', default='base', choices=['base', 'large'], help='which bert model to use')
    parser.add_argument('--max_len', default=128, choices=[32, 64, 128, 256, 512], type=int, help='max sequence length')
    args = parser.parse_args()
    print(args)

    # load bert tokenizer
    tokenizer_name = 'bert-' + args.which_bert + '-' + args.case
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    # read preprocessd files
    if args.stage <= 0:
        train = read_file(args.train)
        test = read_file(args.test)
    
        # obtain tokens for the datasets and check if the size fits bert input
        train_sent_tokens, train_token_lengths = bert_tokens(tokenizer, train, args.max_len)
        test_sent_tokens, test_token_lengths = bert_tokens(tokenizer, test, args.max_len)

        if validate_lengths(train_token_lengths):
            print('Train dataset has sentence(s) which exceeds BERT input length')

        if validate_lengths(test_token_lengths):
            print('Test dataset has sentence(s) which exceeds BERT input length')

        save_tokens(train_sent_tokens, args.out_dir, pad="_no_pad")
        save_tokens(test_sent_tokens, args.out_dir, "test", pad="_no_pad")
        print("Tokens generated successfully")

    # print distribution of lengths
    if args.stage <= 1:
        train_sent_tokens = load_tokens(args.out_dir, pad="_no_pad")
        test_sent_tokens = load_tokens(args.out_dir, "test", pad="_no_pad")

        plot_dist_len(train_sent_tokens, args.out_dir)
        plot_dist_len(test_sent_tokens, args.out_dir, "test")
        print("Distribution plotted")

    # pad tokens
    if args.stage <= 2:
        train_sent_tokens = load_tokens(args.out_dir, pad="_no_pad")
        test_sent_tokens = load_tokens(args.out_dir, "test", pad="_no_pad")

        train_pad_sent_tokens = padding(train_sent_tokens, args.max_len)
        test_pad_sent_tokens = padding(test_sent_tokens, args.max_len)

        f_prefix = "_pad_" + str(args.max_len)
        save_tokens(train_pad_sent_tokens, args.out_dir, pad=f_prefix)
        save_tokens(test_pad_sent_tokens, args.out_dir, "test", pad=f_prefix)
        print("Tokens padded successfully")

    # attention masks
    if args.stage <= 3:
        f_prefix = "_pad_" + str(args.max_len)
        train_pad_sent_tokens = load_tokens(args.out_dir, pad=f_prefix)
        test_pad_sent_tokens = load_tokens(args.out_dir, "test", pad=f_prefix)

        train_sent_att_masks = bert_att_mask(train_pad_sent_tokens)
        test_sent_att_masks = bert_att_mask(test_pad_sent_tokens)

        att_f_prefix = "_att_" + str(args.max_len)
        save_tokens(train_sent_att_masks, args.out_dir, pad=att_f_prefix)
        save_tokens(test_sent_att_masks, args.out_dir, "test", pad=att_f_prefix)
        print("Attention masks generated successfully")

if __name__=='__main__':
    main()
