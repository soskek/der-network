#!/usr/bin/env python
from __future__ import print_function
import argparse
import sys

import chainer
from chainer import cuda
from chainer import serializers
import copy
import time
from datetime import datetime
import json
from links.dern import DERN
from multiprocessing import Pool
import math
import numpy as np
import six


parser = argparse.ArgumentParser()
parser.add_argument('--save-path-model', '-savem', default="./", type=str,
                    help='save path')
parser.add_argument('--data-path', '-dp', default="./", type=str,
                    help='data path')
parser.add_argument('--model-name', '-mn', default="model", type=str,
                    help='model name')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--n-epoch', '--epoch', '-e', default=10, type=int,
                    help='epoch')
parser.add_argument('--n-units', '--units', '-u', default=300, type=int,
                    help='dimension size (# of units)')
parser.add_argument('--per', '-p', default=100, type=int,
                    help='result output timing (per PER iters)')
parser.add_argument('--batchsize', '-b', default=1, type=int,
                    help='size of minibatch')
parser.add_argument('--grad-clip', '-c', default=5, type=float,
                    help='glad clip number')
parser.add_argument('--d-ratio', '--dropout-ratio', '--dropout', '-d', default=0.1, type=float,
                    help='dropout ratio')
parser.add_argument('--init-range', '-ir', default=0.05, type=float,
                    help='init range')
parser.add_argument('--speed-up', '-su', default=-1, type=int,
                    help='highway mode. { -1:none, 1: skip validation at 1-6th epoch, 2: and 7-30th epoch skip validation at even number epoch }')
parser.add_argument('--start-lr', '-lr', default=1e-4, type=float,
                    help='starting learning rate for decayed-sgd')
parser.add_argument('--decay-rate', '-dr', default=1.2, type=float,
                    help='decaying rate (1/DR) of learning rate for decayed-sgd')
parser.add_argument('--shuffle', '-s', default=-1, type=float,
                    help='shuffle partition. data is shuffled every 1/SHUFFLE epoch. (e.g., 5, 10, 0.5)')
parser.add_argument('--save-epoch', '-se', default="1,2,3,4", type=str,
                    help='save timing: e.g "31,34,37" ')
parser.add_argument('--max-ent-id', '-me', default=100, type=int,
                    help='max id number of entities')
parser.add_argument('--n-pool', '--pool', '-np', default=-1, type=int,
                    help='# of parallel pools used for cpu multi processing')
parser.add_argument('--frac-eval', '-fe', default=1, type=int,
                    help='evaluation timing. per 1/FE epoch.')
parser.add_argument('--w2v-path', '-w2v', default=None, type=str,
                    help='w2v path')
parser.add_argument('--load-model', '-lm', default=None, type=str,
                    help='loading pretrained model path')
parser.add_argument('--load-corpus', default=None, type=str,
                    help='load corpus.pkl path')
parser.add_argument('--plus-valid', '-pvalid', default=161, type=int,
                    help='addition for validation dataset')
parser.add_argument('--plus-test', '-ptest', default=137, type=int,
                    help='addition for test dataset')

args = parser.parse_args()
xp = cuda.cupy if args.gpu >= 0 else np
chainer.Function.type_check_enable = False

args.save_path_model = args.save_path_model.rstrip("/")+"/"
args.max_ent_id = args.max_ent_id + 1
args.data_path = args.data_path.rstrip("/")+"/"


def load_processed_dataset(load_corpus):
    return json.load(open(load_corpus))


def setup_model(vocab, args):
    model = DERN(vocab, args)
    optimizer = model.setup_optimizer()
    return model, optimizer


def train_model(args):
    train_data, valid_data, test_data, vocab = load_processed_dataset(args.load_corpus)
    model, optimizer = setup_model(vocab, args)

    print("counting queries.")
    n_query_train_data = sum([len(v_["queries"]) for v_ in train_data])
    print("#train queries =", n_query_train_data)
    n_query_valid_data = sum([len(v_["queries"]) for v_ in valid_data])
    print("#valid queries =", n_query_valid_data)
    print("+#valid queries (skipped) =", args.plus_valid)
    n_query_test_data = sum([len(v_["queries"]) for v_ in test_data])
    print("#test queries =", n_query_test_data)
    print("+#test queries (skipped) =", args.plus_test)
    
    for i_epoch in range(args.n_epoch):
        # Training
        train_epoch(model, optimizer, vocab, train_data, valid_data, i_epoch, args, n_query_valid_data)

        # Evaluate on validation data
        print('evaluate')
        loss_mean, correct_per_, n_choice_per_, sub_correct_per_ = evaluate(valid_data,
                                                                       model,
                                                                       args,
                                                                       n_query_data=n_query_valid_data)
        print('epoch {} validation loss: {:.2f}, accuracy: {:.3f}, +sub: {:.3f} (difficulty:{:.2f})'.format(
            i_epoch+1, loss_mean,
            float(correct_per_/(n_query_valid_data+args.plus_valid)),
            float(sub_correct_per_/(n_query_valid_data+args.plus_valid)),
            (n_choice_per_/n_query_valid_data)))

        # Decay learning rate
        optimizer.lr /= args.decay_rate
        print('learning rate =', optimizer.lr)

        save_name = args.save_path_model + args.model_name + ".%d.%s." % (i_epoch+1, datetime.today().strftime("%Y%m%d.%H%M%S"))
        save(model, optimizer, vocab, save_name, args)

    # Evaluate on test dataset
    print('test')
    loss_mean, correct_per_, n_choice_per_, sub_correct_per_ = evaluate(test_data, model, n_query_test_data)
    print('END test loss: {:.4f}, accuracy: {:.3f}, +sub: {:.3f} (difficulty:{:.2f})'.format(
        loss_mean, float(correct_per_/(n_query_test_data+args.plus_test)),
        float(sub_correct_per_/(n_query_test_data+args.plus_test)), (n_choice_per_/n_query_test_data)))


def save(model, optimizer, vocab, save_name, args):
    serializers.save_npz(save_name+"model", copy.deepcopy(model).to_cpu())
    serializers.save_npz(save_name+"optimizer", optimizer)
    json.dump(vocab, open(save_name+"vocab.json", "w"))
    print('save', save_name)


def make_pool(model, n_pool):
    # make
    pool = Pool(n_pool)
    model.zerograds()
    modelL = [copy.deepcopy(model) for j in six.moves.range(n_pool)]
    return pool, modelL


def train_epoch(model, optimizer, vocab, train_data, valid_data, epoch, args, n_query_valid_data=None):
    model.zerograds()
    pool, modelL = make_pool(model, args.n_pool)
    jump = args.batchsize * args.n_pool
    whole_len = len(train_data)
    sum_loss_data = xp.zeros(())
    
    cur_at = time.time()
    correct_per, sub_correct_per, n_choice_per, processed = 0., 0., 0., 0
    prev_i, processed_i, query_i = 0, 0, 0

    perm = np.random.permutation(len(train_data)).tolist()
    print("Epoch",epoch,"start.")

    for i in six.moves.range(0, len(train_data), jump):
        model.zerograds()

        # Make parallel chunks roughly fairly with respect to data size (almost in propotion to process time)
        idsL = model.make_efficient_chunk(perm[i:i+args.batchsize*args.n_pool], train_data)
        datasL = [[train_data[idx] for idx in ids] for ids in idsL]

        # Solve problems, calculate gradients and sum them from all pools
        for result in pool.imap_unordered(wrapper_solve, zip(modelL, datasL, [True]*args.n_pool)):
            grad_pack, sum_loss_one, n_T, n_choice, n_s = result
            for p in grad_pack:
                if len(p[0]) == 2:
                    getattr(getattr(model, p[0][0]), p[0][1])._grad += p[1]
                else:  # for LSTM
                    getattr(getattr(getattr(model, p[0][0]), p[0][1] ), p[0][2])._grad += p[1]
            sum_loss_data += sum_loss_one
            correct_per += n_T
            sub_correct_per += n_s
            n_choice_per += n_choice

        now_processed = sum([len(train_data[i_]["queries"]) for i_ in perm[i:i+args.batchsize*args.n_pool]])
        processed += now_processed
        processed_i += len(perm[i:i+args.batchsize*args.n_pool])
        query_i += now_processed

        # Update and sync parameters
        optimizer.update()
        for j in six.moves.range(args.n_pool):
            modelL[j].zerograds()
            modelL[j].copyparams(model)

        # Print loss and acc
        if processed_i >= args.per:
            loss_mean = cuda.to_cpu(sum_loss_data) / processed
            now = time.time()
            throuput = processed * 1.0 / (now - cur_at)

            print('iter {}, {} training loss: {:.4f}, accuracy: {:.3f}, +sub: {:.3f} (difficulty:{:.2f}) ({:.2f} iters/sec)\t{}'.format(
                (i + len(perm[i:i+args.batchsize*args.n_pool])), query_i, loss_mean,
                float(correct_per/processed), float(sub_correct_per/processed),
                (n_choice_per/processed), throuput, datetime.today().strftime("%Y/%m/%d,%H:%M:%S") ))
            cur_at = now
            sum_loss_data.fill(0)
            correct_per, n_choice_per, processed, processed_i, sub_correct_per = 0., 0., 0, 0, 0.

        # Evaluate on validation set
        if i-prev_i > whole_len / args.frac_eval:
            prev_i = i
            now = time.time()
            pool.close()
            print('evaluate')
            loss_mean, correct_per_val, n_choice_per_val, sub_correct_per_val = evaluate(valid_data, model, args, n_query_valid_data)
            print('epoch {:.2f} validation loss: {:.4f}, accuracy: {:.3f}, +sub: {:.3f} (difficulty:{:.2f})'.format(
                (epoch+(i*1.0/whole_len)), loss_mean,
                float(correct_per_val/(n_query_valid_data+args.plus_valid)),
                float(sub_correct_per_val/(n_query_valid_data+args.plus_valid)), (n_choice_per_val/n_query_valid_data)))
            cur_at += time.time() - now  # skip time of evaluation

            save_name = args.save_path_model + args.model_name + ".%d.%s." % (epoch+(i*1.0/whole_len), datetime.today().strftime("%Y%m%d.%H%M%S"))
            save(model, optimizer, vocab, save_name, args)
            pool, modelL = make_pool(model, args.n_pool)
    pool.close()

def evaluate(dataset, model, args, n_query_data=None):
    pool, modelL = make_pool(model, args.n_pool)
    correct_per, sub_correct_per, n_choice_per = 0., 0., 0.
    sum_loss_data = xp.zeros(())

    idsL = model.make_efficient_chunk(list(six.moves.range(len(dataset))), dataset)
    all_datasL = [[dataset[idx] for idx in ids] for ids in idsL]

    # Split dataset into some part
    n_ch = len(all_datasL[0])/6+1
    for j in six.moves.range(6):
        datasL = [each_datas[j*n_ch:(j+1)*n_ch] for each_datas in all_datasL]

        for result in pool.imap_unordered(
                wrapper_solve, zip(modelL, datasL, [False]*args.n_pool)):
            sum_loss_one, n_T, n_choice, n_s = result
            sum_loss_data += sum_loss_one
            correct_per += n_T
            sub_correct_per += n_s
            n_choice_per += n_choice
    if n_query_data is None:
        n_query_data = sum([len(v_["queries"]) for v_ in dataset])

    pool.close()
    return cuda.to_cpu(sum_loss_data) / n_query_data, correct_per, n_choice_per, sub_correct_per

def wrapper_solve(args_):
    model, datas, train = args_
    accum_loss = chainer.Variable(model.xp.zeros((), dtype=np.float32), volatile=not train)
    correct_per, sub_correct_per, n_choice_per = 0., 0., 0.
    for data in datas:
        loss_i, TorF, subTorF = model.solve(data, train=train)
        accum_loss += loss_i
        correct_per += TorF
        sub_correct_per += subTorF
        n_choice_per += len(data["entities"]) * len(data["queries"])
    accum_loss_full = accum_loss.data.reshape(())

    if not train:
        return (accum_loss_full, correct_per, n_choice_per, sub_correct_per)
    else:
        model.zerograds()
        accum_loss.backward()
        grad_pack = [(p[0][1:].split("/"), p[1]._grad) for p in model.namedparams()]
        return (grad_pack, accum_loss_full, correct_per, n_choice_per, sub_correct_per)

if __name__ == "__main__":
    print("##### ##### ##### #####")
    print(" ".join(sys.argv))
    print("STARTING TIME:",datetime.today().strftime("%Y/%m/%d %H:%M:%S"))
    print("##### ##### ##### #####")
    for k, v in sorted(args.__dict__.items(), key=lambda x:len(x[0])): print("#",k,":\t",v)
    print("##### ##### ##### #####")

    # Training
    train_model(args)

    print(' ***** E N D ***** ')
