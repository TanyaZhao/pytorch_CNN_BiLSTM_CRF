# coding=utf-8

from __future__ import print_function

import math
import optparse
import os
import pickle
import time
from datetime import datetime
import numpy as np
import random

from data_loader_conll import DataLoaderCoNLL
from data_processing import *
from enums import *
from evaluation import evaluating
from model import BiLSTM_CRF
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3,4,5,6,7'
t = time.time()

optparser = optparse.OptionParser()
optparser.add_option(
    '--name', default='CNN_BiLSTM_CRF',
    help='Model name'
)
optparser.add_option(
    "--train", default="datasets/conll2003/eng.train.bio",
    help="Train set location"
)
optparser.add_option(
    "--dev", default="datasets/conll2003/eng.testa.bio",
    help="Dev set location"
)
optparser.add_option(
    "--test", default="datasets/conll2003/eng.testb.bio",
    help="Test set location"
)
optparser.add_option(
    "--train_subset", default="datasets/conll2003/eng.train.subset",
    help="Test set location"
)
optparser.add_option(
    "--tag_scheme", choices=['iob', 'iobes'], default="iobes",
    help="Tagging scheme (IOB or IOBES)"
)
optparser.add_option(
    "--word_lower", default="1",
    type='int', help="Lowercase words (this will not affect character inputs)"
)
optparser.add_option(
    "--word_threshold", default="3",
    type='int',
    help="Only words with the corresponding threshold larger than or equal to word_threshold will be preserved"
)
optparser.add_option(
    "--digits_to_zeros", default="0",
    type='int', help="Replace digits with 0"
)
optparser.add_option(
    "--char_dim", default="30",
    type='int', help="Char embedding dimension"
)
optparser.add_option(
    '--char_mode', choices=['CNN', 'LSTM'], default='CNN',
    help='char_CNN or char_LSTM'
)
optparser.add_option(
    "--char_lstm_dim", default="30",
    type='int', help="Char embedding LSTM hidden layer size"
)
optparser.add_option(
    "--char_lstm_bidirect", default="1",
    type='int', help="Use a bidirectional LSTM for char embedding"
)
optparser.add_option(
    '--char_cnn_win', default='3',
    type='int', help='CNN win-size for char embedding'
)
optparser.add_option(
    '--char_cnn_dim', default="30",
    type='int', help='Dimensionality of the output of CNN for char embedding, i.e., number of CNN kernels'
)
optparser.add_option(
    "--word_dim", default="100",
    type='int', help="Token embedding dimension"
)
optparser.add_option(
    "--word_lstm_dim", default="200",
    type='int', help="Token embedding LSTM hidden layer size"
)
optparser.add_option(
    "--word_lstm_bidirect", default="1",
    type='int', help="Use a bidirectional LSTM for words"
)
optparser.add_option(
    "--pre_emb", default="resource/embedding/glove.6B.100d.txt",
    help="Location of pre-trained char embeddings"
)
optparser.add_option(
    "--crf", default="1",
    type='int', help="Use CRF (0 to disable)"
)
optparser.add_option(
    "--dropout", default="0.5",
    type='float', help="Droupout on the input (0 = no dropout)"
)
optparser.add_option(
    "--reload", default="0",
    type='int', help="Reload the last saved model"
)
optparser.add_option(
    '--use_gpu', default='1',
    type='int', help='whether or not to use gpu'
)
optparser.add_option(
    '--max_epoch', default='100',
    type='int', help='Maximal number of epochs'
)
optparser.add_option(
    '--mini_batch_size', default='10',
    type='int', help='Mini-batch size'
)
optparser.add_option(
    '--cuda', default='3',
    type='int', help='Index of the cuda device'
)
optparser.add_option(
    '--optimizer', choices=['SGD', 'AdaDelta', 'Adam'], default='SGD',
    help='Optimizer, selected from [SGD, AdaDelta, Adam]'
)
optparser.add_option(
    "--word_cnn_dim", default="400",
    type='int', help="Token embedding CNN hidden layer size"
)
optparser.add_option(
    '--encoder_mode', choices=['pureCNN', 'LSTM', 'biCNN'], default='LSTM',
    help='encoder_CNN or encoder_LSTM'
)


# parse and check all the parameters
opts = optparser.parse_args()[0]
opts_str = "{0}".format(opts)

assert os.path.isfile(opts.train)
assert os.path.isfile(opts.dev)
assert os.path.isfile(opts.test)
assert opts.tag_scheme.lower() in ['iob', 'iobes']
assert opts.word_threshold >= 0
assert opts.char_mode.lower() in ['cnn', 'lstm']
assert opts.word_dim > 0
assert opts.word_lstm_dim > 0
assert opts.dropout >= 0.0 and opts.dropout < 1.0
assert opts.max_epoch > 0
assert opts.mini_batch_size > 0
assert opts.cuda >= 0
assert opts.optimizer.lower() in ['sgd', 'adadelta', 'adam']

name = opts.name
train_set_path = opts.train
dev_set_path = opts.dev
test_set_path = opts.test
train_subset_path = opts.train_subset
label_schema = LabellingSchema.IOB if opts.tag_scheme.lower() == "iob" else LabellingSchema.IOBES
word_to_lower = opts.word_lower == 1
word_frequency_threshold = opts.word_threshold
digits_to_zeros = opts.digits_to_zeros == 1
char_dim = opts.char_dim
char_mode = CharEmbeddingSchema.CNN if opts.char_mode.lower() == "cnn" else CharEmbeddingSchema.LSTM
char_lstm_dim = opts.char_lstm_dim
char_lstm_bidirect = opts.char_lstm_bidirect == 1
char_cnn_win = opts.char_cnn_win
char_cnn_dim = opts.char_cnn_dim
word_dim = opts.word_dim
word_lstm_dim = opts.word_lstm_dim
word_cnn_dim = opts.word_cnn_dim
word_lstm_bidirect = opts.word_lstm_bidirect == 1
prebuilt_embed_path = '' if not os.path.isfile(opts.pre_emb) else opts.pre_emb
use_crf = opts.crf == 1
dropout = opts.dropout
reload = opts.reload == 1
use_gpu = opts.use_gpu == 1 and torch.cuda.is_available()
max_epoch = opts.max_epoch
mini_batch_size = opts.mini_batch_size
encoder_mode = opts.encoder_mode


optimizer_choice = OptimizationMethod.Adam if opts.optimizer.lower() == 'adam' else \
    (OptimizationMethod.AdaDelta if opts.optimizer.lower() == 'adadelta' else OptimizationMethod.SGDWithDecreasingLR)

device_count = torch.cuda.device_count() if use_gpu else 0
assert (not char_mode == CharEmbeddingSchema.LSTM) or char_lstm_dim > 0
assert (not char_mode == CharEmbeddingSchema.CNN) or (char_cnn_win > 0 and char_cnn_dim > 0)
assert (not use_gpu) or opts.cuda < device_count

name = name if reload else "{0}{1}".format(name, datetime.now().strftime('%Y%m%d%H%M%S'))

device_name = "cuda:{0}".format(opts.cuda) if use_gpu else "cpu"
device = torch.device(device_name)
print("Using device {0}".format(device))

if use_gpu:
    torch.cuda.set_device(opts.cuda)

models_path = Constants.Models_Folder
logs_path = Constants.Logs_Folder
eval_path = Constants.Eval_Folder
eval_temp = Constants.Eval_Temp_Folder
eval_script = Constants.Eval_Script

mapping_file = os.path.join(models_path, "{0}.mappings.pkl".format(name))
model_name = os.path.join(models_path, "{0}.model".format(name))

assert (not reload) or os.path.exists(model_name)

if not os.path.isfile(eval_script):
    raise Exception('CoNLL evaluation script not found at "%s"' % eval_script)
if not os.path.exists(eval_temp):
    os.makedirs(eval_temp)
if not os.path.exists(models_path):
    os.makedirs(models_path)
if not os.path.exists(logs_path):
    os.makedirs(logs_path)

# create the mappings
if os.path.exists(mapping_file):
    mappings = pickle.load(open(mapping_file, "rb"))
    prebuilt_word_embedding = load_prebuilt_word_embedding(prebuilt_embed_path, word_dim)
else:
    # build vocab
    mappings, prebuilt_word_embedding = create_mapping_dataset_conll(
        [train_set_path],
        prebuilt_embed_path,
        word_dim,
        label_schema,
        word_to_lower,
        word_frequency_threshold,
        digits_to_zeros)
    pickle.dump(mappings, open(mapping_file, "wb"))
print('Loaded %i pretrained english embeddings.' % len(prebuilt_word_embedding))

tag_to_id = mappings["tag_to_id"]
word_to_id = mappings["word_to_id"]
char_to_id = mappings["char_to_id"]
id_to_tag = mappings["id_to_tag"]

# create dataset loaders
train_set = DataLoaderCoNLL(train_set_path, mappings)
dev_set = DataLoaderCoNLL(dev_set_path, mappings)
test_set = DataLoaderCoNLL(test_set_path, mappings)
train_subset = DataLoaderCoNLL(train_subset_path, mappings)

print("%i / %i sentences in train / test." % (len(train_set), len(test_set)))

if prebuilt_word_embedding is not None:
    word_embeds = np.random.uniform(-np.sqrt(6/word_dim), np.sqrt(6/word_dim), (len(word_to_id), word_dim)) # Kaiming_uniform
    for w in word_to_id.keys():
        if w in prebuilt_word_embedding.keys():
            word_embeds[word_to_id[w], :] = prebuilt_word_embedding[w]
        elif w.lower() in prebuilt_word_embedding.keys():
            word_embeds[word_to_id[w], :] = prebuilt_word_embedding[w.lower()]
else:
    word_embeds = None

print('word_to_id: ', len(word_to_id))
print('tag_to_id: ', len(tag_to_id))

model = BiLSTM_CRF(word_set_size=len(word_to_id),
                   tag_to_id=tag_to_id,
                   word_embedding_dim=word_dim,
                   word_lstm_dim=word_lstm_dim,
                   word_cnn_dim=word_cnn_dim,
                   word_lstm_bidirect=word_lstm_bidirect,
                   pre_word_embeds=word_embeds,
                   encoder_mode=encoder_mode,
                   char_mode=char_mode,
                   char_embedding_dim=char_dim,
                   char_lstm_dim=char_lstm_dim,
                   char_lstm_bidirect=char_lstm_bidirect,
                   char_cnn_win=char_cnn_win,
                   char_cnn_output=char_cnn_dim,
                   char_to_id=char_to_id,
                   use_gpu=use_gpu,
                   dropout=dropout,
                   use_crf=use_crf,
                   )

print(model)

p_count = 0
for parameter in model.parameters():
    if parameter.requires_grad:
        p_count += 1

name_count = 0
param_list = []
for param_name, param in model.named_parameters():
    if param.requires_grad:
        name_count += 1
        print(param_name, "  ", param.size())
        param_list.append(param_name)

print("p_count:{0},name_count:{1}".format(p_count,name_count))

log = str(model)
with open(os.path.join(logs_path, "{0}.important.log".format(name)), "a") as fout:
    fout.write(log)
    fout.write('\n')
    for param in param_list:
        fout.write(param)
        fout.write('\n')
    fout.flush()

if reload:
    last_saved_model = torch.load(model_name, map_location=device_name)
    model.load_state_dict(last_saved_model.state_dict())
    model.use_gpu = use_gpu
if use_gpu:
    model = model.to(device)

# Perf: Adam < AdaDelta < SGD
if optimizer_choice == OptimizationMethod.SGDWithDecreasingLR:
    learning_rate = 0.02
    learning_momentum = 0.9
    print("learning_rate:{0}".format(learning_rate))
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, momentum=learning_momentum)

elif optimizer_choice == OptimizationMethod.Adam:
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
elif optimizer_choice == OptimizationMethod.AdaDelta:
    optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()))

best_dev_results = [-1.0,-1.0,-1.0,-1.0]
best_test_results = [-1.0,-1.0,-1.0,-1.0]
best_train_results = [-1.0,-1.0,-1.0,-1.0]

batch_count = math.ceil(len(train_set) / mini_batch_size)

model.train(True)
for epoch in range(max_epoch):
    train_indecies = np.random.permutation(len(train_set))
    full_logs = []
    if epoch == 0:
        # print(opts_str)
        full_logs.append(opts_str)

    train_time = 0
    for batch_i in range(batch_count):
        start_idx = batch_i * mini_batch_size
        end_idx = min((batch_i + 1) * mini_batch_size, len(train_set))

        mini_batch_idx = train_indecies[start_idx:end_idx]

        sentence_masks, words, chars, tags, \
        sentence_char_lengths, sentence_char_position_map, str_words, unaligned_tags = \
            generate_mini_batch_input(train_set, mini_batch_idx, mappings, char_mode)

        if use_gpu:
            sentence_masks = sentence_masks.to(device)
            words = words.to(device)
            chars = chars.to(device)
            tags = tags.to(device)
            sentence_char_lengths = sentence_char_lengths.to(device)

        start_train = datetime.now()

        model.zero_grad()

        neg_log_likelihood = model.neg_log_likelihood(words, sentence_masks, tags, chars,
                                                      sentence_char_lengths, sentence_char_position_map, device)
        neg_log_likelihood.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        end_train = datetime.now()
        train_time += (end_train - start_train).total_seconds()

        loss = neg_log_likelihood.data.item()
        log = "epoch {0} batch {1}/{2} loss {3}".format(epoch + 1, batch_i + 1, batch_count, loss)
        # print(log)
        full_logs.append(log)

    print("train_time:{}".format(train_time))

    model.eval()
    with torch.no_grad():
        best_train_results, new_train_results, _, _ = evaluating(model, train_subset, best_train_results, name, mappings, char_mode, use_gpu,
                                                  device, mini_batch_size)
        best_dev_results, new_dev_results, save, _ = evaluating(model, dev_set, best_dev_results, name, mappings, char_mode, use_gpu, device, mini_batch_size)

        best_test_results, new_test_results, _, test_time = evaluating(model, test_set, best_test_results, name, mappings, char_mode, use_gpu,device, mini_batch_size)

        if save:
            torch.save(model, model_name)
            test_with_best_dev = new_test_results

        log = "Epoch {0}: [{1:.5f}, {2:.5f}, {3:.5f}, {4:.5f}, {5:.5f}], [accuracy,precision,recall]:[{6:.5f},{7:.5f},{8:.5f}]".format(epoch + 1, best_test_results[0], new_test_results[0],
                                                                                                                                       best_dev_results[0], new_dev_results[0], test_with_best_dev[0],
                                                                                                                                       test_with_best_dev[1], test_with_best_dev[2], test_with_best_dev[3])
        print(log)
        print("test_time:{}".format(test_time))

        full_logs.append(log)
        full_logs.append('\n')

        with open(os.path.join(logs_path, "{0}.full.log".format(name)), "a") as fout:
            fout.write('\n'.join(full_logs))
            fout.flush()

        with open(os.path.join(logs_path, "{0}.important.log".format(name)), "a") as fout:
            fout.write(log)
            fout.write('\n')
            fout.flush()

    model.train(True)

    if optimizer_choice == OptimizationMethod.SGDWithDecreasingLR:
        adjust_learning_rate(optimizer, lr=learning_rate / (1 + 0.05 * (epoch + 1)))

print(time.time() - t)
