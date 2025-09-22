import os
import time
import numpy as np
import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
from tqdm import tqdm
import time
import torch.nn.functional as F
from contrastive import contrastive


def compute_score_with_logits(logits, labels):
    logits = torch.argmax(logits, 1)
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores

def compute_sample_difficulty(pred):
    probs = torch.softmax(pred, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
    return entropy


def map_confidence_to_shuffle_ratio(confidence, H, ratio=7, beta=4.0):
    s = int(torch.mean(ratio + beta*(confidence/H)))
    if s>14:
        s=14
    elif s<0:
        s=0
    return s

def adjusted_labels(labels, head_classes, tail_classes, mid_classes, delta_head=0.13, delta_tail=0.00, delta_mid=0.00):
    """
    labels: (batch_size, num_classes) -> one-hot 形式
    head_classes: 头部类别索引
    tail_classes: 尾部类别索引
    delta_head: 头部类别的 delta
    delta_tail: 尾部类别的 delta
    delta_mid: 其他类别（中部）的 delta
    """
    batch_size, num_classes = labels.size()

    gt_labels = torch.argmax(labels, dim=1)  # (batch_size,)


    head_mask = torch.isin(gt_labels, head_classes)
    tail_mask = torch.isin(gt_labels, tail_classes)
    mid_mask = torch.isin(gt_labels, mid_classes)

    head_nonzero_mask = labels[head_mask] != 0
    tail_nonzero_mask = labels[tail_mask] != 0
    mid_nonzero_mask = labels[mid_mask] != 0

    # adjusted_labels
    adjusted_labels = labels.clone()

    # delta / num_classes
    adjusted_labels[head_mask] =  delta_head / num_classes
    adjusted_labels[tail_mask] = delta_tail / num_classes
    adjusted_labels[mid_mask] = delta_mid / num_classes

    #1 - delta + delta / num_classes
    batch_indices, class_indices = (labels != 0).nonzero(as_tuple=True)


    head_batch_indices = batch_indices[head_mask[batch_indices]]
    head_class_indices = class_indices[head_mask[batch_indices]]
    tail_batch_indices = batch_indices[tail_mask[batch_indices]]
    tail_class_indices = class_indices[tail_mask[batch_indices]]
    mid_batch_indices = batch_indices[mid_mask[batch_indices]]
    mid_class_indices = class_indices[mid_mask[batch_indices]]

    adjusted_labels[head_batch_indices, head_class_indices] = 1 - delta_head + delta_head / num_classes
    adjusted_labels[tail_batch_indices, tail_class_indices] = 1 - delta_tail + delta_tail / num_classes
    adjusted_labels[mid_batch_indices, mid_class_indices] = 1 - delta_mid + delta_mid / num_classes

    return adjusted_labels

def train(model, qmodel, train_loader, eval_loader, qid2type, args):

    epochs = args.epochs
    output = args.output
    batchsize = args.batch_size
    run_eval = args.eval_each_epoch

    npyname = 'label_counts.npy'
    q_ynp = np.load(npyname)
    q_y = torch.from_numpy(q_ynp).cuda()
    sorted_indices = torch.argsort(q_y, descending=True)

    # head/mid/tail
    num_classes = len(sorted_indices)
    head_threshold = int(0.4 * num_classes)
    mid_threshold = int(0.7 * num_classes)

    head_classes = sorted_indices[:head_threshold]
    mid_classes = sorted_indices[head_threshold:mid_threshold]
    tail_classes = sorted_indices[mid_threshold:]

    optim = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    logger = utils.Logger(os.path.join(output, 'log.txt'))
    total_step = 0
    best_eval_score = 0
    best_epoch = 0

    bce = nn.BCELoss()

    for epoch in range(epochs):

        loss_trk = []
        acc_trk = []
        train_score = 0
        total_loss = 0

        loader = tqdm(train_loader, ncols=0)
        t = time.time()

        for v, q, a0, qid in loader:
            total_step+=1

            v = v.cuda()
            q = q.cuda()
            #a= a0.cuda()
            a = adjusted_labels(a0, head_classes, tail_classes, mid_classes)
            a = a.cuda()

            optim.zero_grad()

            q_logits = qmodel(q)
            _, gt_labels = torch.max(a, dim=1)
            q_loss = F.binary_cross_entropy_with_logits(q_logits, a, reduction='mean')
            confidence = q_logits[torch.arange(len(gt_labels)), gt_labels]
            h = compute_sample_difficulty(q_logits)
            shuffle_ratios = map_confidence_to_shuffle_ratio(confidence, h)

            pred = model(v, q, ratio=shuffle_ratios)


            score = compute_score_with_logits(pred, a.data)
            batch_score = score.sum(dim=1).mean()
            sum_score = score.sum()
            train_score += sum_score

            loss_bce = F.binary_cross_entropy_with_logits(pred, a, reduction='none')
            batch_loss_bce = loss_bce.sum(dim=1).mean()


            sum_loss_bce = loss_bce.sum()
            loss =  q_loss + batch_loss_bce
            total_loss += sum_loss_bce
            #print(loss)
            loss.backward()
            optim.step()


            fmt = '{:.4f}'.format
            loss_trk.append(batch_loss_bce)
            acc_trk.append(batch_score)


            loader.set_postfix(loss=fmt(sum(loss_trk)/len(loss_trk)), acc=fmt(100 * sum(acc_trk)/len(acc_trk)))


        train_score = 100 * train_score / len(train_loader.dataset)
        total_loss /= len(train_loader.dataset)

        logger.write('Epoch %d, time: %.2f' % (epoch + 1, time.time() - t))
        logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))


        if run_eval:
            model.train(False)
            results = evaluate(model, eval_loader, qid2type)
            results["epoch"] = epoch
            results["step"] = total_step
            results["train_loss"] = total_loss
            results["train_score"] = train_score

            model.train(True)

            eval_score = results["score"]
            bound = results["upper_bound"]
            yn = results['score_yesno']
            other = results['score_other']
            num = results['score_number']
            logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))
            logger.write('\tyn score: %.2f other score: %.2f num score: %.2f' % (100 * yn, 100 * other, 100 * num))
            main_eval_score = eval_score

            if main_eval_score > best_eval_score:
                model_path = os.path.join(output, 'model.pth')
                best_epoch = epoch
                torch.save(model.state_dict(), model_path)
                best_eval_score = main_eval_score
    #model_path = os.path.join(output, 'model.pth')
    #torch.save(model.state_dict(), model_path)
    print("best_epoch:%d best_eval_score:%f" %(best_epoch, best_eval_score))


def evaluate(model, dataloader, qid2type):
    score = 0
    upper_bound = 0
    score_yesno = 0
    score_number = 0
    score_other = 0
    total_yesno = 0
    total_number = 0
    total_other = 0

    for v, q, a, qids in tqdm(dataloader, ncols=100, total=len(dataloader), desc="eval"):
        v = Variable(v, requires_grad=False).cuda()
        q = Variable(q, requires_grad=False).cuda()
        pred = model(v, q)
        batch_score = compute_score_with_logits(pred, a.cuda()).cpu().numpy().sum(1)
        score += batch_score.sum()
        upper_bound += (a.max(1)[0]).sum()
        qids = qids.detach().cpu().int().numpy()
        for j in range(len(qids)):
            qid = qids[j]
            typ = qid2type[str(qid)]
            if typ == 'yes/no':
                score_yesno += batch_score[j]
                total_yesno += 1
            elif typ == 'other':
                score_other += batch_score[j]
                total_other += 1
            elif typ == 'number':
                score_number += batch_score[j]
                total_number += 1
            else:
                print('Hahahahahahahahahahaha')

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    score_yesno /= total_yesno
    score_other /= total_other
    score_number /= total_number

    results = dict(
        score=score,
        upper_bound=upper_bound,
        score_yesno=score_yesno,
        score_other=score_other,
        score_number=score_number,
    )
    return results