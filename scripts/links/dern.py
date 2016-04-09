#!/usr/bin/env python
from __future__ import print_function
import sys

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from collections import defaultdict
from collections import OrderedDict
import numpy as np
import six


class DERN(chainer.Chain):

    def __init__(self, vocab, args):
        def get_initialW_X(shape):
            return np.random.normal(0, (2.0/(sum(shape)))**0.5, shape).astype(np.float32)

        super(DERN, self).__init__(
            # Word Embedding
            embed=L.EmbedID(len(vocab), args.n_units),

            # bi-LSTMs
            f_LSTM=L.LSTM(args.n_units, args.n_units),  # for article
            b_LSTM=L.LSTM(args.n_units, args.n_units),
            Q_f_LSTM=L.LSTM(args.n_units, args.n_units),  # for query
            Q_b_LSTM=L.LSTM(args.n_units, args.n_units),

            # Matrices and vectors
            W_hd=L.Linear(4*args.n_units, args.n_units, initialW=get_initialW_X((args.n_units, 4*args.n_units))),
            W_dm=L.Linear(args.n_units, args.n_units, initialW=get_initialW_X((args.n_units, args.n_units))),
            m=L.Linear(args.n_units, 1, initialW=get_initialW_X((1, args.n_units))),
            W_hq=L.Linear(4 * args.n_units, args.n_units, initialW=get_initialW_X((args.n_units, 4*args.n_units))),
            W_hu=L.Linear(4 * args.n_units, args.n_units, initialW=get_initialW_X((args.n_units, 4*args.n_units))),
            W_dv=L.Linear(args.n_units, args.n_units, initialW=get_initialW_X((args.n_units, args.n_units))),
            W_dx=L.Linear(args.n_units, args.n_units, initialW=get_initialW_X((args.n_units, args.n_units))),
            W_dxQ=L.Linear(args.n_units, args.n_units, initialW=get_initialW_X((args.n_units, args.n_units))),

            b_v2=L.Linear(1, args.n_units, initialW=get_initialW_X((args.n_units, 1)))
        )

        self.args = args
        self.n_vocab = len(vocab)
        self.n_units = args.n_units
        self.dropout_ratio = args.d_ratio

        self.PH_id = vocab["@placeholder"]
        self.eos_id = vocab["<eos>"]
        self.bos_id = vocab["<bos>"]
        self.boq_id = vocab["<boq>"]
        self.BOQ_tok_batch = self.xp.array([self.boq_id], dtype=np.int32)
        self.NULL_id = vocab["NULL_tok"]
        self.NULL_tok = self.xp.array(self.NULL_id, dtype=np.int32)

        self.initialize_additionally()

    def read_w2v(self, w2iD, w2v_file):
        """Load pre-trained word embeddings.

        w2v_file (str): a file name of word embeddings.
        the format is as follows:
        WORD_1 dim1 dim2 dim3 ... dim300
        WORD_2 dim1 dim2 dim3 ... dim300
        ...
        WORD_N dim1 dim2 dim3 ... dim300
        """
        wordS = set(w2iD.keys())
        print('LOADING WORD VECTOR...')
        word_count = 0
        embed_W = self.embed.W.data
        for l in open(w2v_file):
            sp = l.strip().split()
            word = sp[0].lower()
            if word not in wordS:
                continue

            embed_W[w2iD[word]] = np.array([float(x) for x in sp[1:self.n_units+1]], dtype=np.float32) / 2.
            wordS -= set([word])
            word_count += 1
            if word_count >= self.n_vocab:
                break
        print('%d words DONE.\n' % word_count)

    def get_initialW_X(self, shape):
        return np.random.normal(0, (2.0/(sum(shape)))**0.5, shape).astype(np.float32)

    def initialize_additionally(self):
        self.embed.W.data[:] = np.random.uniform(-self.args.init_range, self.args.init_range,
                                                 (self.n_vocab, self.n_units)).astype(np.float32)

        self.f_LSTM.upward.W.data[:] = self.get_initialW_X((4*self.n_units, self.n_units))
        self.f_LSTM.lateral.W.data[:] = self.get_initialW_X((4*self.n_units, self.n_units))
        self.b_LSTM.upward.W.data[:] = self.get_initialW_X((4*self.n_units, self.n_units))
        self.b_LSTM.lateral.W.data[:] = self.get_initialW_X((4*self.n_units, self.n_units))

        self.Q_f_LSTM.upward.W.data[:] = self.get_initialW_X((4*self.n_units, self.n_units))
        self.Q_f_LSTM.lateral.W.data[:] = self.get_initialW_X((4*self.n_units, self.n_units))
        self.Q_b_LSTM.upward.W.data[:] = self.get_initialW_X((4*self.n_units, self.n_units))
        self.Q_b_LSTM.lateral.W.data[:] = self.get_initialW_X((4*self.n_units, self.n_units))

    def encode_tokens(self, x_datas, i2sD, train=True):
        # Embed, dropout, split into each token (batchsize=1)
        h0L = list(F.split_axis(
            F.dropout(
                self.embed(chainer.Variable(self.xp.array(x_datas, dtype=np.int32), volatile=not train)),
                ratio=self.dropout_ratio, train=train), len(x_datas), axis=0))

        # Replace embedding with dynamic entity representation
        for i in i2sD.keys():
            h0L[i] = self.W_dx(i2sD[i])

        # LSTM. forward order
        forward_outL = []
        self.f_LSTM.reset_state()
        for h0 in h0L:
            state = self.f_LSTM(h0)
            forward_outL.append(state)

        # LSTM. backward order
        backward_outL = []
        self.b_LSTM.reset_state()
        for h0 in reversed(h0L):
            state = self.b_LSTM(h0)
            backward_outL.append(state)

        return forward_outL, backward_outL

    def encode_query(self, x_datas, i2sD, train=True):
        h0L = list(F.split_axis(
            F.dropout(
                self.embed(chainer.Variable(self.xp.array(x_datas, dtype=np.int32), volatile=not train)),
                ratio=self.dropout_ratio, train=train), len(x_datas), axis=0))

        for i in i2sD.keys():
            h0L[i] = self.W_dxQ(i2sD[i])

        placeholder_idx = x_datas.index(self.PH_id)

        # forward
        self.Q_f_LSTM.reset_state()
        for h0 in h0L[:placeholder_idx+1]:
            state = self.Q_f_LSTM(h0)
        forward_out = state
        for h0 in h0L[placeholder_idx+1:]:
            state = self.Q_f_LSTM(h0)
        forward_endout = state

        # backward
        self.Q_b_LSTM.reset_state()
        for h0 in reversed(h0L[placeholder_idx:]):
            state = self.Q_b_LSTM(h0)
        backward_out = state
        for h0 in reversed(h0L[:placeholder_idx]):
            state = self.Q_b_LSTM(h0)
        backward_endout = state

        concat_h = F.concat([forward_out, backward_out, forward_endout, backward_endout], axis=1)
        return self.W_hu(concat_h), self.W_hq(concat_h)

    def concat_outputs(self, e_s, iL, forward_outL, backward_outL, train=True):

        if len(iL) == 1:  # Target appears only ``once'' in the sentence
            four = [forward_outL[iL[0]], backward_outL[len(forward_outL)-1-iL[0]], forward_outL[-1], backward_outL[-1]]
        else:  # Target appears over once in the sentence. Target output is the average of all occurances.
            four = [[], [], forward_outL[-1], backward_outL[-1]]
            for i in iL[1:]:
                four[0].append(forward_outL[i])
                four[1].append(backward_outL[len(forward_outL)-1-i])
            four[0] = sum(four[0])/len(iL)
            four[1] = sum(four[1])/len(iL)

        concat_h = F.concat(four, axis=1)
        return concat_h

    def encode_context(self, x_datas, i2sD, e2iLD, train=True):
        forward_outL, backward_outL = self.encode_tokens(x_datas, i2sD, train=train)
        concat_h_L = []

        for iL, e_s in zip(e2iLD.values(), [i2sD[iL_[0]] for iL_ in e2iLD.values()]):
            concat_h = self.concat_outputs(e_s, iL, forward_outL, backward_outL, train=train)
            concat_h_L.append(concat_h)
        return concat_h_L

    def attention_history(self, dL, cue, train=True):
        D = F.concat(dL, axis=0)
        D, Cue = F.broadcast(D, cue)
        S = self.m(F.tanh(self.W_dm(D) + Cue))
        S = F.softmax(F.reshape(S, (1, len(dL))))
        pre_v = F.matmul(S, D)
        return pre_v

    def initialize_entities(self, entities, max_entnum, train=True):
        e2sD = {}
        old2newD = {}

        if train:
            news = self.xp.random.randint(0, max_entnum, len(entities))
        else:
            news = entities

        new_e_L = []
        for new, entity in zip(news, entities):
            old2newD[entity] = int(new)
            new_e_L.append(new)

        es_L = F.split_axis(
            self.embed(chainer.Variable(self.xp.array(new_e_L, dtype=np.int32), volatile=not train)),
            len(new_e_L), axis=0)
        if len(new_e_L) <= 1:
            es_L = [es_L]
        for new_e, es in zip(new_e_L, es_L):
            e2sD[new_e] = es

        return old2newD, e2sD

    def reload_sentence(self, sentence, old2newD):
        return [token if token not in old2newD else old2newD[token] for token in sentence]

    def reload_sentences(self, old_sentences, old2newD):
        return [self.reload_sentence(sentence, old2newD) for sentence in old_sentences]

    def make_heuristic_vec(self, e_occur_L, train=True):
        hot_vec = self.xp.zeros((len(e_occur_L), 1), dtype=np.float32)
        for i, e_o in enumerate(e_occur_L):
            if e_o:
                hot_vec[i] = 1.0
        return self.b_v2(chainer.Variable(hot_vec, volatile=not train))

    def predict_answer(self, u_Dq, v_eDq, e_occur_L, train=True):
        v_eDq2 = v_eDq + self.make_heuristic_vec(e_occur_L, train=train)
        score = F.matmul(u_Dq, v_eDq2, transb=True)
        return score

    def solve(self, docD, train=True):
        old2newD, e2sD = self.initialize_entities(docD["entities"], self.args.max_ent_id, train=train)
        e2dLD = dict((e, [s]) for (e, s) in e2sD.items())
        sentences = self.reload_sentences(docD["sentences"], old2newD)

        for sent in sentences:
            i2sD = OrderedDict()
            e2iLD = defaultdict(list)
            for i, token in enumerate(sent):
                if token in e2sD:
                    i2sD[i] = e2sD[token]
                    e2iLD[token].append(i)
            if not i2sD:  # skip sentences without any entities
                continue
            e2iLD = OrderedDict(e2iLD)

            concat_h_L = self.encode_context(sent, i2sD, e2iLD, train=train)
            for e, concat_h in zip(e2iLD.keys(), concat_h_L):
                e2dLD[e].append(F.tanh(self.W_hd(concat_h)))
                e2sD[e] = F.max(F.concat([e2sD[e], e2dLD[e][-1]], axis=0), axis=0, keepdims=True)

        EPS = sys.float_info.epsilon
        accum_loss_doc, TorFs, subTorFs = 0, 0, 0

        for query, answer in zip(docD["queries"], docD["answers"]):
            query = self.reload_sentence(query, old2newD)
            answer = old2newD[int(answer)]
            i2sD = dict([(i, e2sD[token]) for i, token in enumerate(query) if token in e2sD])
            u_Dq, q = self.encode_query(query, i2sD, train=train)
            eL, sL = zip(*list(e2sD.items()))
            pre_vL = [self.attention_history(e2dLD[e], q, train=train) for e in eL]
            v_eDq = self.W_dv(F.concat(pre_vL, axis=0))
            answer_idx = eL.index(answer)

            p = self.predict_answer(u_Dq, v_eDq, [True if token in query else False for token in eL], train=train) + EPS
            t = chainer.Variable(self.xp.array([answer_idx]).astype(np.int32), volatile=not train)
            accum_loss_doc += F.softmax_cross_entropy(p, t)

            p_data = p.data[0, :]
            max_idx = self.xp.argmax(p_data)
            TorFs += (max_idx == answer_idx)
            if max_idx != answer_idx:
                for sub_ans in [k for k, e in enumerate(eL) if e in query]:
                    p_data[sub_ans] = -10000000
                subTorFs += (self.xp.argmax(p_data) == answer_idx)

        return accum_loss_doc, TorFs, subTorFs
    
    def make_efficient_chunk(self, ids, dataset_):
        allD = OrderedDict((idx, len(dataset_[idx]["sentences"]))
                           for idx in sorted(ids, key=lambda x:len(dataset_[x]["sentences"])))
        batchD = dict((i, {"size":0, "data":[]}) for i in six.moves.range(self.args.n_pool))
        for i in six.moves.range(len(allD)//self.args.n_pool+1):
            for key in sorted(batchD.keys(), key=lambda x:batchD[x]["size"]):
                if not allD:
                    break
                pop_ = allD.popitem()
                batchD[key]["data"].append(pop_[0])
                batchD[key]["size"] += pop_[1]
        return [v["data"] for v in batchD.values()]

    def setup_optimizer(self):
        optimizer = optimizers.RMSpropGraves(
            lr=self.args.start_lr, alpha=0.95, momentum=0.9, eps=1e-08)
        optimizer.setup(self)
        optimizer.add_hook(chainer.optimizer.GradientClipping(self.args.grad_clip))
        return optimizer
