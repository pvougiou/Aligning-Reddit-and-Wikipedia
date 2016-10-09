"""
Copyright 2016 Pavlos Vougiouklis

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import json
import codecs
import numpy as np
import h5py
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--reddit', default='../Data/Reddit/Sequences.txt')
parser.add_argument('--redditSequences', default='../Data/Reddit/Sequences.json')
parser.add_argument('--wikipedia', default='../Data/Wikipedia/Summaries.txt')
parser.add_argument('--wikipediaSentences', default='../Data/Wikipedia/Sentences.json')
parser.add_argument('--padding', action='store_true', default=True)
parser.add_argument('--alignedSentences', default=20)
parser.add_argument('--redditOutput', default='../Aligned-Dataset/reddit.h5')
parser.add_argument('--wikipediaOutput', default='../Aligned-Dataset/wikipedia.h5')
parser.add_argument('--dictionary', default='../Aligned-Dataset/dictionary.json')
parser.add_argument('--validate', type=float, default=0.1)
parser.add_argument('--test', type=float, default=0.1)
parser.add_argument('--encoding', default='utf-8')
args = parser.parse_args()


id2word = {}
word2id = {}
word2count = {}
vocab_length = 0
id2element = {}
element2id = {}
elements_length = 0


with codecs.open(args.wikipedia, 'r', 'utf-8') as wikipedia:
    wikipedia_dataset = wikipedia.read().split()
    wikipedia.close()

    with codecs.open(args.reddit, 'r', 'utf-8') as reddit:
        reddit_dataset = reddit.read().split()
        reddit.close()
        dataset = wikipedia_dataset + reddit_dataset
        
        wikipedia_size = len(wikipedia_dataset)
        reddit_size = len(reddit_dataset)
        size = len(dataset)
        print ('Total number of words included in the Wikipedia dataset: %d' % (wikipedia_size))
        print ('Total number of words included in the reddit dataset: %d' % (reddit_size))
        print ('Total number of words included in the aggregated dataset: %d' % (size))

        vocab_length = vocab_length + 1
        word2id['NaN'.decode('utf-8')] = vocab_length
        id2word[vocab_length] = 'NaN'.decode('utf-8')
        
        # Remove infrequent words from the sequences of comments.

        for i in range(0, wikipedia_size):
             if wikipedia_dataset[i] not in word2count:
                word2count[wikipedia_dataset[i]] = 1
             else:
                 word2count[wikipedia_dataset[i]] = word2count[wikipedia_dataset[i]] + 1
                 if wikipedia_dataset[i] not in word2id:
                     vocab_length = vocab_length + 1
                     word2id[wikipedia_dataset[i]] = vocab_length
                     id2word[vocab_length] = wikipedia_dataset[i]
        print ('Size of the vocabulary after processing the Wikipedia dataset: %d' % (vocab_length))

        for i in range(0, reddit_size):
            if reddit_dataset[i] not in word2count:
                word2count[reddit_dataset[i]] = 1
            elif word2count[reddit_dataset[i]] <= 2:
                word2count[reddit_dataset[i]] = word2count[reddit_dataset[i]] + 1
            else:
                word2count[reddit_dataset[i]] = word2count[reddit_dataset[i]] + 1
                if reddit_dataset[i] not in word2id:
                    vocab_length = vocab_length + 1
                    word2id[reddit_dataset[i]] = vocab_length
                    id2word[vocab_length] = reddit_dataset[i]

if args.padding:
    vocab_length = vocab_length + 1
    word2id['<PAD>'.decode('utf-8')] = vocab_length
    id2word[vocab_length] = '<PAD>'.decode('utf-8')
    
print ('Size of the vocabulary: %d' % (vocab_length))


# Load Wikipedia sentences along with their corresponding lengths.

with open (args.wikipediaSentences, 'r') as f:
    sentences = json.load(f, 'utf-8')
    f.close()

wiki_sentences = sentences['wiki_sentences']
wiki_elements = sentences['wiki_elements']
wiki_sentences_length = sentences['wiki_sentences_length']
num_wiki_sentences = len(wiki_sentences)

max_tokens = 0
for i in range(0, num_wiki_sentences):
    if len(wiki_sentences[i]) > max_tokens:
        max_tokens = len(wiki_sentences[i])

print ('Max sentence length is: %d' % (max_tokens))

for i in range(0, len(wiki_sentences)):
    if wiki_elements[i] not in element2id:
        elements_length = elements_length + 1
        element2id[wiki_elements[i]] = elements_length
        id2element[elements_length] = wiki_elements[i] 


# Load reddit sequences.

with open (args.redditSequences, 'r') as f:
    sequences = json.load(f, 'utf-8')
    f.close()

tmpSequences = []
tmpElements = []
sequences_length = []
num_sequences = 0

for element in sequences:
    for j in range(0, len(sequences[element])):
        tmpSequences.append(sequences[element][j])
        tmpElements.append(element)
        sequences_length.append(len(sequences[element][j].split()))
        num_sequences = num_sequences + 1

reddit_lengths = np.asarray(sequences_length)
reddit_mean = np.mean(reddit_lengths)
reddit_std = np.std(reddit_lengths)
reddit_maximum = np.amax(reddit_lengths)

# Disregard sequences whose length exceeds a specific threshold.
threshold = reddit_mean + 1.0 * reddit_std

deleted = 0
tmpSequencesLength = sequences_length[:]
for i in range(0, num_sequences):
    if tmpSequencesLength[i] > threshold:
        del tmpSequences[i - deleted]
        del tmpElements[i - deleted]
        del sequences_length[i - deleted]
        deleted = deleted + 1
        
reddit_lengths = np.asarray(sequences_length)
reddit_mean = np.mean(reddit_lengths)
reddit_std = np.std(reddit_lengths)
reddit_maximum = np.amax(reddit_lengths)

print ('Mean sequence length is: %d' % (reddit_mean))
print ('Max sequence length is: %d' % (reddit_maximum))
print ('Standard deviation of the lengths of the sampled sequences is: %d' % (reddit_std))        

reddit_validate_sequences = np.floor(args.validate * len(tmpSequences))
reddit_test_sequences = np.floor(args.test * len(tmpSequences))
reddit_train_sequences = len(tmpSequences) - reddit_validate_sequences - reddit_test_sequences

aligned_train_dataset = []
aligned_validate_dataset = []
aligned_test_dataset = []


# The flagElement corresponds to the position of the last sentence that has been aligned with an already processed sequence of comments.
flagElement = {}

for key in element2id:    
    flagElement[key] = 0
totalSequences = [0, 0, 0]

while len(tmpSequences) > 0:

    sequence = np.random.randint(0, len(tmpSequences))

    if totalSequences[1] < reddit_validate_sequences and totalSequences[2] < reddit_test_sequences:
        random = np.random.randint(0, 3)
    elif totalSequences[1] >= reddit_validate_sequences and totalSequences[2] < reddit_test_sequences:
        random = np.random.randint(0, 2)
        if random == 1:
            random = random + 1
    elif totalSequences[2] >= reddit_test_sequences and totalSequences[1] < reddit_validate_sequences:
        random = np.random.randint(0, 2)
    else:
        random = 0
        
    if random == 1 and totalSequences[1] < reddit_validate_sequences:
        aligned_validate_dataset.append({'sentences': [], 'sequence': tmpSequences[sequence].split()})
        totalSequences[1] = totalSequences[1] + 1
        find = 0
        while wiki_elements[find] != tmpElements[sequence]:
            find = find + 1
        included = 0
        while included < args.alignedSentences:
            if find + flagElement[tmpElements[sequence]] + included < len(wiki_sentences):
                if wiki_elements[find + flagElement[tmpElements[sequence]] + included] == tmpElements[sequence]:
                    aligned_validate_dataset[len(aligned_validate_dataset) - 1]['sentences'].append(wiki_sentences[find + flagElement[tmpElements[sequence]] + included])
                    included = included + 1
                else:
                    flagElement[tmpElements[sequence]] = 0
            else:
                flagElement[tmpElements[sequence]] = 0
                print('Resetting end-flag for the Element: %s...' % (tmpElements[sequence]))
        flagElement[tmpElements[sequence]] = flagElement[tmpElements[sequence]] + included
        tmpSequences.pop(sequence)
        tmpElements.pop(sequence)
        sequences_length.pop(sequence)

    elif random == 2 and totalSequences[2] < reddit_test_sequences:
        aligned_test_dataset.append({'sentences': [], 'sequence': tmpSequences[sequence].split()})
        totalSequences[2] = totalSequences[2] + 1
        find = 0
        while wiki_elements[find] != tmpElements[sequence]:
            find = find + 1
        included = 0
        while included < args.alignedSentences:
            if find + flagElement[tmpElements[sequence]] + included < len(wiki_sentences):
                if wiki_elements[find + flagElement[tmpElements[sequence]] + included] == tmpElements[sequence]:
                    aligned_test_dataset[len(aligned_test_dataset) - 1]['sentences'].append(wiki_sentences[find + flagElement[tmpElements[sequence]] + included])
                    included = included + 1
                else:
                    flagElement[tmpElements[sequence]] = 0
            else:
                flagElement[tmpElements[sequence]] = 0
                print('Resetting end-flag for the Element: %s...' % (tmpElements[sequence]))
        flagElement[tmpElements[sequence]] = flagElement[tmpElements[sequence]] + included
        tmpSequences.pop(sequence)
        tmpElements.pop(sequence)
        sequences_length.pop(sequence)

    else:
        #reddit_train_dataset = reddit_train_dataset + tmpSequences[sequence]
        aligned_train_dataset.append({'sentences': [], 'sequence': tmpSequences[sequence].split()})
        totalSequences[0] = totalSequences[0] + 1
        find = 0
        while wiki_elements[find] != tmpElements[sequence]:
            find = find + 1
        included = 0
        while included < args.alignedSentences:
            if find + flagElement[tmpElements[sequence]] + included < len(wiki_sentences):
                if wiki_elements[find + flagElement[tmpElements[sequence]] + included] == tmpElements[sequence]:
                    aligned_train_dataset[len(aligned_train_dataset) - 1]['sentences'].append(wiki_sentences[find + flagElement[tmpElements[sequence]] + included])
                    included = included + 1
                else:
                    flagElement[tmpElements[sequence]] = 0
            else:
                flagElement[tmpElements[sequence]] = 0
                print('Resetting end-flag for the Element: %s...' % (tmpElements[sequence]))
        flagElement[tmpElements[sequence]] = flagElement[tmpElements[sequence]] + included
        tmpSequences.pop(sequence)
        tmpElements.pop(sequence)
        sequences_length.pop(sequence)
        
        
aligned_reddit_train = np.zeros((len(aligned_train_dataset), reddit_maximum), dtype=np.uint32)
aligned_reddit_validate = np.zeros((len(aligned_validate_dataset), reddit_maximum), dtype=np.uint32)
aligned_reddit_test = np.zeros((len(aligned_test_dataset), reddit_maximum), dtype=np.uint32)

aligned_wiki_train = np.zeros((args.alignedSentences * len(aligned_train_dataset), max_tokens), dtype=np.uint32)
aligned_wiki_validate = np.zeros((args.alignedSentences * len(aligned_validate_dataset), max_tokens), dtype=np.uint32)
aligned_wiki_test = np.zeros((args.alignedSentences * len(aligned_test_dataset), max_tokens), dtype=np.uint32)

for i in range(0, len(aligned_train_dataset)):
    for j in range(0, len(aligned_train_dataset[i]['sequence'])):
        if aligned_train_dataset[i]['sequence'][j] in word2id:
            aligned_reddit_train[i][j] = word2id[aligned_train_dataset[i]['sequence'][j]]
        else:
            aligned_reddit_train[i][j] = word2id['NaN']
    for z in range(len(aligned_train_dataset[i]['sequence']), reddit_maximum):
                aligned_reddit_train[i][z] = word2id['<PAD>']
    for j in range(0, len(aligned_train_dataset[i]['sentences'])):
        for z in range(0, len(aligned_train_dataset[i]['sentences'][j])):
            if aligned_train_dataset[i]['sentences'][j][z] in word2id:
                aligned_wiki_train[args.alignedSentences * i + j][z] = word2id[aligned_train_dataset[i]['sentences'][j][z]]
            else:
                aligned_wiki_train[args.alignedSentences * i + j][z] = word2id['NaN']
        for t in range(len(aligned_train_dataset[i]['sentences'][j]), max_tokens):
                aligned_wiki_train[args.alignedSentences * i + j][t] = word2id['<PAD>']

for i in range(0, len(aligned_validate_dataset)):
    for j in range(0, len(aligned_validate_dataset[i]['sequence'])):
        if aligned_validate_dataset[i]['sequence'][j] in word2id:
            aligned_reddit_validate[i][j] = word2id[aligned_validate_dataset[i]['sequence'][j]]
        else:
            aligned_reddit_validate[i][j] = word2id['NaN']
    for z in range(len(aligned_validate_dataset[i]['sequence']), reddit_maximum):
                aligned_reddit_validate[i][z] = word2id['<PAD>']
    for j in range(0, len(aligned_validate_dataset[i]['sentences'])):
        for z in range(0, len(aligned_validate_dataset[i]['sentences'][j])):
            if aligned_validate_dataset[i]['sentences'][j][z] in word2id:
                aligned_wiki_validate[args.alignedSentences * i + j][z] = word2id[aligned_validate_dataset[i]['sentences'][j][z]]
            else:
                aligned_wiki_validate[args.alignedSentences * i + j][z] = word2id['NaN']
        for t in range(len(aligned_validate_dataset[i]['sentences'][j]), max_tokens):
                aligned_wiki_validate[args.alignedSentences * i + j][t] = word2id['<PAD>']
                       
for i in range(0, len(aligned_test_dataset)):
    for j in range(0, len(aligned_test_dataset[i]['sequence'])):
        if aligned_test_dataset[i]['sequence'][j] in word2id:
            aligned_reddit_test[i][j] = word2id[aligned_test_dataset[i]['sequence'][j]]
        else:
            aligned_reddit_test[i][j] = word2id['NaN']
    for z in range(len(aligned_test_dataset[i]['sequence']), reddit_maximum):
                aligned_reddit_test[i][z] = word2id['<PAD>']
    for j in range(0, len(aligned_test_dataset[i]['sentences'])):
        for z in range(0, len(aligned_test_dataset[i]['sentences'][j])):
            if aligned_test_dataset[i]['sentences'][j][z] in word2id:
                aligned_wiki_test[args.alignedSentences * i + j][z] = word2id[aligned_test_dataset[i]['sentences'][j][z]]
            else:
                aligned_wiki_test[args.alignedSentences * i + j][z] = word2id['NaN']
        for t in range(len(aligned_test_dataset[i]['sentences'][j]), max_tokens):
                aligned_wiki_test[args.alignedSentences * i + j][t] = word2id['<PAD>']


with h5py.File(args.wikipediaOutput, 'w') as f:
    f.create_dataset('train', data=aligned_wiki_train)
    f.create_dataset('validate', data=aligned_wiki_validate)
    f.create_dataset('test', data=aligned_wiki_test)
    
with h5py.File(args.redditOutput, 'w') as f:
    f.create_dataset('train', data=aligned_reddit_train)
    f.create_dataset('validate', data=aligned_reddit_validate)
    f.create_dataset('test', data=aligned_reddit_test)


dictionary = {'word2id': word2id, 'id2word': id2word}
with open(args.dictionary, 'w') as f:
    json.dump(dictionary, f, encoding='utf-8')
