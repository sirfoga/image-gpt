import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from imageio import imwrite
from scipy.special import softmax

from src.model import model
from src.utils import iter_data, count_parameters, AttrDict


IMAGE_GPT_CONFIGS = {
    's': {
        'n_ctx': 32 * 32,  # 32-pixel square image
        'n_embd': 512,
        'n_head': 8,
        'n_layer': 24,
        'n_vocab': 256,  # grayscale image
        'bert': True,
        'bert_mask_prob': 0.15,
        'clf': False,  # no classification fine-tuning
    }
}

H_CONFIG = {
    'data_path': './data/',
    'ckpt_path': './ckpt/',
    'color_cluster_path': './quants/',
    'save_dir': './results/',
    'n_sub_batch': 8,
    'n_gpu': 1,  # we are poor
    'n_px': 32,  # square image size in pixels
    'eval': False,  # not needed (for now)
    'sample': False,  # not needed (for now)
    'seed': 42,
    'train_epochs': 30
}


def parse_arguments():
    parser = argparse.ArgumentParser()

    # data and I/O
    parser.add_argument("--data_path", type=str, default="/root/downloads/imagenet")
    parser.add_argument("--ckpt_path", type=str, default="/root/downloads/model.ckpt-1000000")
    parser.add_argument("--color_cluster_path", type=str, default="/root/downloads/kmeans_centers.npy")
    parser.add_argument("--save_dir", type=str, default="/root/save/")

    # model
    parser.add_argument("--n_embd", type=int, default=512)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_layer", type=int, default=24)
    parser.add_argument("--n_px", type=int, default=32, help="image height or width in pixels")
    parser.add_argument("--n_vocab", type=int, default=512, help="possible values for each pixel")

    parser.add_argument("--bert", action="store_true", help="use the bert objective (defaut: autoregressive)")
    parser.add_argument("--bert_mask_prob", type=float, default=0.15)
    parser.add_argument("--clf", action="store_true", help="add a learnable classification head")

    # parallelism
    parser.add_argument("--n_sub_batch", type=int, default=8, help="per-gpu batch size")
    parser.add_argument("--n_gpu", type=int, default=8, help="number of gpus to distribute training across")

    # mode
    parser.add_argument("--eval", action="store_true", help="evaluates the model, requires a checkpoint and dataset")
    parser.add_argument("--sample", action="store_true", help="samples from the model, requires a checkpoint and clusters")

    # reproducibility
    parser.add_argument("--seed", type=int, default=42, help="seed for random, np, tf")

    args = parser.parse_args()
    print("input args:\n", json.dumps(vars(args), indent=4, separators=(",", ":")))
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def load_data(data_path):
    trX = np.load(f'{data_path}_trX.npy')
    trY = np.load(f'{data_path}_trY.npy')
    vaX = np.load(f'{data_path}_vaX.npy')
    vaY = np.load(f'{data_path}_vaY.npy')
    teX = np.load(f'{data_path}_teX.npy')
    teY = np.load(f'{data_path}_teY.npy')
    return (trX, trY), (vaX, vaY), (teX, teY)


def set_hparams(args):
    return AttrDict(
        dict(
            n_ctx=args.n_px * args.n_px,
            n_embd=args.n_embd,
            n_head=args.n_head,
            n_layer=args.n_layer,
            n_vocab=args.n_vocab,
            bert=args.bert,
            bert_mask_prob=args.bert_mask_prob,
            clf=args.clf,
        )
    )


def reduce_mean(gen_loss, tot_loss, n_gpu):
    with tf.device("/gpu:0"):
        for i in range(1, n_gpu):
            gen_loss[0] += gen_loss[i]  # reduce
            tot_loss[0] += tot_loss[i]

        gen_loss[0] /= n_gpu  # mean
        tot_loss[0] /= n_gpu


def create_model(x, y=None, n_gpu=1, hparams=None):
    trainable_params = None
    for i in range(n_gpu):
        with tf.device("/gpu:%d" % i):
            reuse = True  # i != 0  # reuse params just in GPU > 0

            if y:
                results = model(hparams, x[i], y[i], reuse=reuse)
            else:
                results = model(hparams, x[i], reuse=reuse)

            gen_logits.append(results["gen_logits"])
            gen_loss.append(results["gen_loss"])

            tot_loss.append(results["gen_loss"])

            if i == 0:
                trainable_params = tf.trainable_variables()
                print("trainable parameters:", count_parameters())

    return trainable_params, gen_logits, gen_loss, tot_loss


def evaluate(sess, evX, evY, X, Y, gen_loss, n_batch, desc, permute=False):
    metrics = []
    for xmb, ymb in iter_data(evX, evY, n_batch=n_batch, truncate=True, verbose=True):
        metrics.append(sess.run([gen_loss[0]], {X: xmb, Y: ymb}))
    eval_gen_loss,  = [np.mean(m) for m in zip(*metrics)]
    print(f"{desc} gen: {eval_gen_loss:.4f}")


# naive sampler without caching
def sample(sess, X, gen_logits, n_sub_batch, n_gpu, n_px, n_vocab, clusters, save_dir):
    samples = np.zeros([n_gpu * n_sub_batch, n_px * n_px], dtype=np.int32)

    for i in range(n_px * n_px):
        np_gen_logits = sess.run(gen_logits, {X: samples})
        for j in range(n_gpu):
            p = softmax(np_gen_logits[j][:, i, :], axis=-1)  # logits to probas
            for k in range(n_sub_batch):
                c = np.random.choice(n_vocab, p=p[k])  # choose based on probas
                samples[j * n_sub_batch + k, i] = c

    # dequantize
    samples = [np.reshape(np.rint(127.5 * (clusters[s] + 1.0)), [32, 32, 3]).astype(np.uint8) for s in samples]

    # write to png
    for i in range(n_gpu * n_sub_batch):
        imwrite(f"{args.save_dir}/sample_{i}.png", samples[i])


def main(args):
    set_seed(args.seed)

    n_batch = args.n_sub_batch * args.n_gpu

    if args.data_path.endswith("cifar10"):
        n_class = 10
    elif args.data_path.endswith("imagenet"):
        n_class = 1000
    else:
        n_class = 0  # dataset not supported

    X = tf.placeholder(tf.int32, [n_batch, args.n_px * args.n_px])
    Y = tf.placeholder(tf.float32, [n_batch, n_class])

    x = tf.split(X, args.n_gpu, 0)
    y = tf.split(Y, args.n_gpu, 0)

    hparams = set_hparams(args)
    trainable_params, gen_logits, gen_loss, tot_loss = create_model(x, y, args.n_gpu, hparams)
    reduce_mean(gen_loss, tot_loss, args.n_gpu)

    saver = tf.train.Saver(var_list=[tp for tp in trainable_params if not 'clf' in tp.name])
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        sess.run(tf.global_variables_initializer())

        saver.restore(sess, args.ckpt_path)

        if args.eval:
            (trX, trY), (vaX, vaY), (teX, teY) = load_data(args.data_path)
            evaluate(sess, trX[:len(vaX)], trY[:len(vaY)], X, Y, gen_loss, n_batch, "train")
            evaluate(sess, vaX, vaY, X, Y, gen_loss, n_batch, "valid")
            evaluate(sess, teX, teY, X, Y, gen_loss, n_batch, "test")

        if args.sample:
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            clusters = np.load(args.color_cluster_path)
            sample(sess, X, gen_logits, args.n_sub_batch, args.n_gpu, args.n_px, args.n_vocab, clusters, args.save_dir)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
