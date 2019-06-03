import tensorflow as tf
import numpy as np
import os, argparse, time
from cxf_zh_NER_BiLSTM_CRF.model import BiLSTM_CRF
from cxf_zh_NER_BiLSTM_CRF.myutils import str2bool, get_logger, get_entity
from cxf_zh_NER_BiLSTM_CRF.mydata import read_corpus, read_dictionary, tag2label, random_embedding
from cxf_zh_NER_BiLSTM_CRF.newsPlaceExtract import FileTools_no_labels
from collections import OrderedDict
from datetime import datetime

#
start_time = datetime.now()

# Session configuration
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.2  # need ~700MB GPU memory

# hyperparameters
parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')
parser.add_argument('--train_data', type=str, default='data_path', help='train data osdata')
parser.add_argument('--test_data', type=str, default='data_path', help='test data osdata')
parser.add_argument('--batch_size', type=int, default=64, help='#sample of each minibatch')
parser.add_argument('--epoch', type=int, default=40, help='#epoch of training')
parser.add_argument('--hidden_dim', type=int, default=300, help='#dim of hidden state')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--CRF', type=str2bool, default=True, help='use CRF at the top layer. if False, use Softmax')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
parser.add_argument('--update_embedding', type=str2bool, default=False, help='update embedding during training')
parser.add_argument('--pretrain_embedding', type=str, default='pretrain',
                    help='use pretrained char embedding or init it randomly')
parser.add_argument('--embedding_dim', type=int, default=300, help='random init char embedding_dim')
parser.add_argument('--shuffle', type=str2bool, default=True, help='shuffle training data before each epoch')
parser.add_argument('--mode', type=str, default='test', help='train/test/demo')
parser.add_argument('--demo_model', type=str, default='1558598108', help='model for test and demo')
args = parser.parse_args()

# get char embeddings
word2id = OrderedDict()
embeddings = []
if args.pretrain_embedding == 'random':
    word2id = read_dictionary(os.path.join('.', args.train_data, 'word2id.pkl'))
    embeddings = random_embedding(word2id, args.embedding_dim)
    with open('tmp/tmp.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(word2id.keys()))
else:
    embedding_path = 'data_path/baike_charvect.txt'
    with open(embedding_path, 'r', encoding='utf-8') as f:
        i = 0
        for line in f:
            line = line.strip()
            lst = line.split(' ')
            if len(lst) != 301:
                print(line)
                continue
            word2id[lst[0]] = i
            i += 1
            embeddings.append((lst[1:]))
    embeddings = np.float32(embeddings)

# read corpus and get training data
if args.mode == 'train':
    train_path = os.path.join('.', args.train_data, 'train_data')
    test_path = os.path.join('.', args.test_data, 'test_data')
    train_data = read_corpus(train_path)
    test_data = read_corpus(test_path);
    test_size = len(test_data)

if args.mode == 'test':
    #input_file = os.path.join('.', args.test_data, 'News_app1_20190523.xls')
    #input_file = os.path.join('.', args.test_data, 'News_app2_20190523.xls')
    #input_file = os.path.join('.', args.test_data, 'News_epaper1_20190523.xls')
    #input_file = os.path.join('.', args.test_data, 'News_epaper2_20190523.xls')
    #input_file = os.path.join('.', args.test_data, 'News_epaper3_20190523.xls')
    #input_file = os.path.join('.', args.test_data, 'News_news1_20190523.xls')
    #input_file = os.path.join('.', args.test_data, 'News_news2_20190523.xls')
    #input_file = os.path.join('.', args.test_data, 'News_news3_20190523.xls')
    #input_file = os.path.join('.', args.test_data, 'News_wechat_20190523.xls')
    input_file = os.path.join('.', args.test_data, 'News_weibo_20190523.xls')
    # 只用于平时测试的新闻
    # input_file = FLAGS.data_dir + '/' + 'news_test.xls'
    df = FileTools_no_labels.read_xls_data(input_file)

# paths setting
paths = {}
timestamp = str(int(time.time())) if args.mode == 'train' else args.demo_model
output_path = os.path.join('.', args.train_data + "_save", timestamp)
if not os.path.exists(output_path): os.makedirs(output_path)
summary_path = os.path.join(output_path, "summaries")
paths['summary_path'] = summary_path
if not os.path.exists(summary_path): os.makedirs(summary_path)
model_path = os.path.join(output_path, "checkpoints/")
if not os.path.exists(model_path): os.makedirs(model_path)
ckpt_prefix = os.path.join(model_path, "model")
paths['model_path'] = ckpt_prefix
result_path = os.path.join(output_path, "results")
paths['result_path'] = result_path
if not os.path.exists(result_path): os.makedirs(result_path)
log_path = os.path.join(result_path, "log.txt")
paths['log_path'] = log_path
get_logger(log_path).info(str(args))

# training model
if args.mode == 'train':
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()

    # hyperparameters-tuning, split train/dev
    # dev_data = train_data[:5000]; dev_size = len(dev_data)
    # train_data = train_data[5000:]; train_size = len(train_data)
    # print("train data: {0}\ndev data: {1}".format(train_size, dev_size))
    # model.train(train=train_data, dev=dev_data)

    # train model on the whole training data
    print("train data: {}".format(len(train_data)))
    model.train(train=train_data, dev=test_data)  # use test_data as the dev_data to see overfitting phenomena

# testing model
elif args.mode == 'test':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    # print("test data: {}".format(test_size))
    # model.test(test_data)
    news_list = []
    for index, row in df.iterrows():
        news_list.append(row['text'])

    model.test_news(news_list)

    print('Program running time ', datetime.now() - start_time)


# demo
elif args.mode == 'demo':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        print('============= demo =============')
        saver.restore(sess, ckpt_file)
        while (1):
            print('Please input your sentence:')
            demo_sent = input()
            if demo_sent == '' or demo_sent.isspace():
                print('See you next time!')
                break
            else:
                demo_sent = list(demo_sent.strip())
                demo_data = [(demo_sent, ['O'] * len(demo_sent))]
                tag = model.demo_one(sess, demo_data)
                PER, LOC, ORG = get_entity(tag, demo_sent)
                print('PER: {}\nLOC: {}\nORG: {}'.format(PER, LOC, ORG))
