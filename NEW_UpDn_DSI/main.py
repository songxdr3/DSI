import argparse
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from torch.utils.data import DataLoader

from datasetgqa import Dictionary, VQAFeatureDataset
import base_model
import q_model
from train import train
import utils
import click

def parse_args():
    parser = argparse.ArgumentParser("Train the BottomUpTopDown model with a de-biasing method")

    # Arguments we added
    parser.add_argument('--cache_features', default=False, help="Cache image features in RAM. Makes things much faster"
                        "especially if the filesystem is slow, but requires at least 48gb of RAM")
    parser.add_argument('--dataset', default='gqaood', choices=["v2", "cpv2", "cpv1"], help="Run on VQA-2.0 instead of VQA-CP 2.0")
    parser.add_argument('--eval_each_epoch', default=True, help="Evaluate every epoch, instead of at the end")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--output', type=str, default='exp1')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--load_checkpoint_path', type=str, default=None)
    args = parser.parse_args()
    return args


def main():

    args = parse_args()
    dataset=args.dataset
    args.output=os.path.join('logs',args.output)

    if not os.path.isdir(args.output):
        utils.create_dir(args.output)
    # else:
    #     if click.confirm('Exp directory already exists in {}. Erase?'
    #                              .format(args.output, default=False)):
    #         os.system('rm -r ' + args.output)
    #         utils.create_dir(args.output)
    #
    #     else:


    if dataset=='cpv1':
        dictionary = Dictionary.load_from_file('../data/vqa/dictionary_v1.pkl')
    elif dataset=='cpv2' or dataset=='v2':
        dictionary = Dictionary.load_from_file('../data/vqa/dictionary.pkl')
    elif dataset == 'gqaood':
        dictionary = Dictionary.load_from_file('/opt/data/private/sxd/data/gqa/dictionary_gqaood.pkl')

    print("Building train dataset...")


    train_dset = VQAFeatureDataset('train', dictionary, dataset=dataset,
                               cache_image_features=args.cache_features)

    print("Building test dataset...")
    eval_dset = VQAFeatureDataset('val', dictionary, dataset=dataset,
                                  cache_image_features=args.cache_features)

    # Build the model using the original constructor
    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(train_dset, args.num_hid).cuda()
    qmodel = getattr(q_model, 'build_Qmodel')(train_dset, args.num_hid).cuda()

    if dataset=='cpv1':
        model.w_emb.init_embedding('../data/vqa/glove6b_init_300d_v1.npy')

    elif dataset=='cpv2' or dataset=='v2':
        model.w_emb.init_embedding('../data/vqa/glove6b_init_300d.npy')
    elif dataset == 'gqaood':
        model.w_emb.init_embedding('/opt/data/private/sxd/data/gqa/glove6b_init_300d.npy')

    if dataset != 'gqaood':
        with open('util/qid2type_%s.json'%args.dataset,'r') as f:
            qid2type=json.load(f)
    else:
        qid2type = None


    if args.load_checkpoint_path is not None:
        path = args.load_checkpoint_path + 'model.pth'
        print(path)
        model_state_dict = torch.load(path)
        model.load_state_dict(model_state_dict)
        print('Model loaded!')

    model=model.cuda()

    batch_size = args.batch_size

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=6)
    eval_loader = DataLoader(eval_dset, batch_size, shuffle=False, num_workers=2)

    print("Starting training...")
    train(model, qmodel, train_loader, eval_loader, qid2type, args)

if __name__ == '__main__':
    main()
