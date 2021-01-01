import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import re
import matplotlib.pyplot as plt
import argparse
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def str2bool(val):
    return True if "True" else False

def read_file(f_name):
    dataset = pd.read_csv(f_name)
    dataset.dropna(inplace=True)
    return dataset

def plot_dist(dataset, out_dir):
    # class dist
    sns.countplot(x=dataset.values[:, -1])
    plt.savefig(out_dir + '/class_count.png', dpi=480)
    plt.close()
    
    # length dist
    lengths = [ len(tweet.split()) for tweet in dataset['text']]
    sns.countplot(x=lengths)
    plt.savefig(out_dir + '/lengths.png', dpi=480)
    plt.close()

def emoji2text(string):
    happy_emojis = [":-\)",":\)",":-\]",":\]",":-3",":3",":->",
                    ":>","8-\)","8\)",":-\}",":\}",":o\)",":c\)",
                    ":^\)","=\]","=\)",":-D",":D","8-D","8D",
                    "x-D","xD","X-D","XD","=D","=3","B^D",
                    ":-\)\)",":'-\)",":'\)",":-O",":O",":-o",":o",
                    ":-0","8-0",":-\*",":\*",":x",";-\)",";\)",
                    "\*-\)","\*\)",";-\]",";\]",";^\)",":-,",";D",
                    ":-P",":P","X-P","XP","x-p","xp",":-p",
                    ":p",":-b",":b","d:","=p",">:P","O:-\)","O:\)",
                    "0:-3","0:3","0:-\)","0:\)","0;^\)",">:-\)",
                    ">:\)","\}:-\)","\}:\)","3:-\)","3:\)",">;\)"
                    ">:3",";3","\|;-\)","#-\)"]

    sad_emojis = [":-\(",":\(",":-c",":c",":-<",":<",":-\[",":\[",
                  ":-\|\|",">:\[",":\{",":@",":\(",";\(",":'-\)",":'\)",
                  "D-':","D:<","D:","D8","D;","D=","DX",">:0",
                  ":-/",":/",":-\.",">:/","=/",":L","=L",":S",":-\|",":\|"
                  "\|-O",":E",":-###\.\.",":###\.\.","',:-\|","',:-l"]

    neutral_emojis = [":$","://\)","://3",":-X",":X",":-#",":#",
                      ":-&",":&","%-\)","%\)"]

    for emoji in happy_emojis:
        string = re.sub(emoji, " happy ", string)

    for emoji in sad_emojis:
        string = re.sub(emoji, " sad ", string)

    for emoji in neutral_emojis:
        string = re.sub(emoji, " neutral ", string)

    string = re.sub("<3", "heart", string)
    return string

def cleaner(data):
    dataset = data.copy()
    for row in dataset.index:
        # remove mentions
        dataset.loc[row, 'text'] = re.sub(r"@[a-zA-Z0-9_]* ", "mention", dataset.loc[row, 'text'])
        
        # remove HTML garbages
        dataset.loc[row, 'text'] = re.sub(r"&\w+;", " ", dataset.loc[row, 'text'])
        dataset.loc[row, 'text'] = re.sub(r"https?:\/\/([^,\s]+)", " ", dataset.loc[row, 'text'])

        # substitute emojis
        dataset.loc[row, 'text'] = emoji2text(dataset.loc[row, 'text'])

        # remove '#' keywords
        dataset.loc[row, 'text'] = re.sub("#", "", dataset.loc[row, 'text'])

        # remove words with exaggerated letters with only 2 repetitions at max
        dataset.loc[row, 'text'] = re.sub(r"(.)\1+", r"\1\1", dataset.loc[row, 'text'])

        # lower case all
        dataset.loc[row, 'text'] = dataset.loc[row, 'text'].lower()

    return dataset

def stopword_removal(data):
    dataset = data.copy()

    stop_words = set(stopwords.words('english'))
    for row in dataset.index:
        sent = dataset.loc[row, 'text']
        word_tokens = word_tokenize(sent)
        cleaner_words = [word for word in word_tokens if not word in stop_words]
        dataset.loc[row, 'text'] = " ".join(cleaner_words) if len(cleaner_words) else dataset.loc[row, 'text']

    return dataset

def save_labels(dataset, out_dir, name="train"):
    label_dic = {'negative': 0, 'neutral': 1, 'positive': 2}
    labels = dataset.values[:, -1]
    model_labels = np.array([label_dic[label] for label in labels])
    with open(out_dir + '/' + name + '_labels.pkl', 'wb') as f:
        pickle.dump(model_labels, f)
        
def main():
    parser = argparse.ArgumentParser(description='Preprocessing', add_help=True)
    parser.add_argument('train', help='Train csv')
    parser.add_argument('test', help='Test csv')
    parser.add_argument('out_dir', help='output directory')
    parser.add_argument('--stage', type=int, help='resume preprocessing stage')
    parser.add_argument('--rem_stop', default=True, choices=[True, False], type=str2bool, help='stopword processed data required?')
    args = parser.parse_args()
    print(args)

    # read original dataset
    train = read_file(args.train)
    test = read_file(args.test)

    # plot the distributions and labels generation
    if args.stage <= 0:
        plot_dist(train, args.out_dir)
        save_labels(train, args.out_dir)
        save_labels(test, args.out_dir, "test")
        print("Plots generated")

    # remove or replace garbage information
    if args.stage <= 1:
        train = cleaner(train)
        test = cleaner(test)

        train.to_csv(args.out_dir + "/preprocessed_train.csv", header=True, index=False)
        test.to_csv(args.out_dir + "/preprocessed_test.csv", header=True, index=False)

        print("Garbage removal complete")

    # remove stop words
    if args.stage <=2 and args.rem_stop:

        train = read_file(args.out_dir + "/preprocessed_train.csv")
        test = read_file(args.out_dir + "/preprocessed_test.csv")

        train = stopword_removal(train)
        test = stopword_removal(test)

        train.to_csv(args.out_dir + "/stop_word_preprocessed_train.csv", header=True, index=False)
        test.to_csv(args.out_dir + "/stop_word_preprocessed_test.csv", header=True, index=False)

        print("Stop words removal complete")

    print("Finished")

if __name__=='__main__':
    main()
