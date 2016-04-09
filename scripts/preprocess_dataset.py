#!/usr/bin/env python
from __future__ import print_function
import argparse

import copy
from glob import glob
import json


parser = argparse.ArgumentParser()
parser.add_argument('--data-path', '--data-dir', '-dir', default="./", type=str,
                    help='data dir path. parent directory of test, validation, training')
parser.add_argument('--save-file', '--save-path', '-save', default="./", type=str,
                    help='save file path')
parser.add_argument('--max-ent-id', '-me', default=100, type=int,
                    help='max id number of entities')
args = parser.parse_args()


def initialize_vocab():
    # Prepare dataset (preliminary download dataset by ./download.py)
    global vocab
    global bos_id, eos_id
    vocab = {}
    for i in range(600):
        vocab["@entity%d" % i] = i
    vocab["@placeholder"] = len(vocab)

    vocab["<eos>"] = len(vocab)
    vocab["<bos>"] = len(vocab)
    vocab["<boq>"] = len(vocab)
    vocab["NULL_tok"] = len(vocab)
    bos_id = vocab["<bos>"]
    eos_id = vocab["<eos>"]


def preprocess_dataset(args):
    initialize_vocab()

    print("start train data")
    train_data = load_data(list_up_files(args.data_path, "training"), args)
    print("start valid data")
    valid_data = load_data(list_up_files(args.data_path, "validation"), args)
    print("start test data")
    test_data = load_data(list_up_files(args.data_path, "test"), args)

    print('#vocab =', len(vocab))
    print('#train =', len(train_data))
    print(' #train q =', sum([len(v["queries"]) for v in train_data]))
    print('#valid =', len(valid_data))
    print(' #valid q =', sum([len(v["queries"]) for v in valid_data]))
    print('#test  =', len(test_data))
    print(' #test  q =', sum([len(v["queries"]) for v in test_data]))
    print("SAVING DATASET", args.save_file)

    json.dump( (train_data, valid_data, test_data, vocab), open(args.save_file, "w"))
    print("END", args.save_file)


def renew_entities(docD):
    old2newD = {}
    newsentL = []
    for i,e in enumerate(docD["entities"]):
        old2newD[e] = i
    for sent in docD["sentences"]:
        newsentL.append([old2newD[tok] if tok in old2newD else tok
                         for tok in sent])
    docD["sentences"] = newsentL
    docD["queries"] = [[old2newD[tok] if tok in old2newD else tok
                        for tok in docD["queries"][0]]]
    docD["entities"] = list(old2newD.values())
    docD["answers"] = [old2newD[docD["answers"][0]] if docD["answers"][0] in old2newD else docD["answers"][0]]
    return docD

def load_doc(filename, args, url2idxD={}):
    global vocab
    docD = { "filenames":[filename], "sentences":[], "entities":[] }
    for i,l in enumerate(open(filename)):
        if i == 0:
            docD["url"] = l.strip()
        elif i == 2:
            if docD["url"] not in url2idxD:
                for sent in l.strip().split("<eos>"):
                    tokenL = []
                    for token in sent.split():
                        if token not in vocab:
                            vocab[token] = len(vocab)
                        tokenL.append( vocab[token] )
                    if tokenL:
                        tokenL.insert( 0, bos_id )
                        tokenL.append( eos_id )
                        docD["sentences"].append( tokenL )
        elif i == 4:
            tokenL = []
            for token in l.strip().split():
                if token not in vocab:
                    vocab[token] = len(vocab)
                tokenL.append( vocab[token] )
            tokenL.insert( 0, bos_id )
            tokenL.append( eos_id )
            docD["queries"] = [tokenL]
        elif i == 6:
            token = l.strip()
            if token not in vocab:
                vocab[token] = len(vocab)
            docD["answers"] = [vocab[token]]
        elif i >= 8:
            docD["entities"].append( int(l.strip().split(":")[0].split("entity")[-1]) )
    if len(docD["entities"]) >= args.max_ent_id - 1 or not docD["answers"][0] in docD["entities"]:
        return None
    docD = renew_entities(docD)
    return docD


def load_data(filelist, args, doc_limit=10000000):
    dataset = []
    url2idxD = {}
    n_file = len(filelist)
    per = n_file / 10
    for i, filename in enumerate(filelist):
        if i % per == 0:
            print("\t",i,"/",n_file)
        docD = load_doc(filename, args, url2idxD)
        if docD:
            if docD["url"] in url2idxD:
                dataset[url2idxD[docD["url"]]]["queries"].append(docD["queries"][0])
                dataset[url2idxD[docD["url"]]]["answers"].append(docD["answers"][0])
            else:
                url2idxD[docD["url"]] = len(dataset)
                dataset.append( docD )
        # shortcut
        if len(dataset) >= doc_limit:
            break
    return dataset


def list_up_files(data_path, data_type):
    filelist = glob(data_path.rstrip("/") + "/" + data_type + "/*")
    return sorted(filelist)


if __name__ == "__main__":
    preprocess_dataset(args)
