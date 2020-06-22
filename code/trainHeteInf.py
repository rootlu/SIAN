# coding: utf-8
# author: lu yf
# create date: 2019-07-13 16:49

import torch
import torch.nn as nn
import numpy as np
import time
import random
import torch.utils.data
import argparse
import os
import datetime
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, precision_recall_curve, accuracy_score
from HeteInf import HeteInf
from Evaluation import Evaluation
from Logging import Logging
from DataUtil import DataUtil
from tqdm import tqdm


def train(model, train_loader, valid_loader, test_loader, optimizer, device):
    """
    train model
    :param model: model object
    :param train_loader: train data loader
    :param valid_loader: validation data loader
    :param test_loader: test data loader
    :param optimizer: optimizer
    :param device: cpu or gpu
    :return: train_loss, valid_loss and test_loss
    """
    model.train()
    train_loss = 0.0
    valid_loss = 0.0
    test_loss = 0.0
    criterion = nn.BCELoss()
    # training
    for i, data in tqdm(enumerate(train_loader, 0)):
        # t0 = time.time()
        batch_user, batch_item, batch_label, batch_act = data['user'].long().to(device), \
                                                         data['item'].long().to(device), \
                                                         data['label'].float().to(device), \
                                                         data['act']
        optimizer.zero_grad()
        output = model(batch_user, batch_item, batch_act)
        loss = criterion(output, batch_label)
        loss.backward(retain_graph=False)
        optimizer.step()
        train_loss += loss.data

    # # validation
    # for i, data in enumerate(valid_loader, 0):
    #     batch_user, batch_item, batch_label, batch_act = data['user'].long().to(device), \
    #                                                      data['item'].long().to(device), \
    #                                                      data['label'].float().to(device), \
    #                                                      data['act']
    #     output = model(batch_user, batch_item, batch_act)
    #     loss = criterion(output, batch_label)
    #     valid_loss += loss.data
    # # test
    # for i, data in enumerate(test_loader, 0):
    #     batch_user, batch_item, batch_label, batch_act = data['user'].long().to(device), \
    #                                                      data['item'].long().to(device), \
    #                                                      data['label'].float().to(device), \
    #                                                      data['act']
    #     output = model(batch_user, batch_item, batch_act)
    #     loss = criterion(output, batch_label)
    #     test_loss += loss.data

    return train_loss, valid_loss, test_loss


def evaluate_clf(model, eval_loader, device, thr=None, return_best_thr=False):
    """
    evaluation for classification
    :param model:
    :param eval_loader:
    :param device:
    :param thr:
    :param return_best_thr:
    :return:
    """
    loss = 0.0
    criterion = nn.BCELoss()
    model.eval()
    y_true, y_pred, y_score = [], [], []
    for i, data in enumerate(eval_loader, 0):
        batch_user, batch_item, batch_label, batch_act = data['user'].long().to(device), \
                                                         data['item'].long().to(device), \
                                                         data['label'].float().to(device), \
                                                         data['act']
        output = model(batch_user, batch_item, batch_act)
        loss += criterion(output, batch_label).data

        y_true += batch_label.data.tolist()
        tmp_pred = torch.where(output.data.cpu() > 0.5, torch.full_like(output.data.cpu(), 1), output.data.cpu())
        y_pred += torch.where(tmp_pred <= 0.5, torch.full_like(output.data.cpu(), 0), tmp_pred).tolist()
        y_score += output.data.tolist()

    model.train()

    if thr is not None:
        y_score = np.array(y_score)
        y_pred = np.zeros_like(y_score)
        y_pred[y_score > thr] = 1

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    auc = roc_auc_score(y_true, y_score)
    acc = accuracy_score(y_true, y_pred)

    if return_best_thr:
        precs, recs, thrs = precision_recall_curve(y_true, y_score)
        f1s = 2 * precs * recs / (precs + recs)
        f1s = f1s[:-1]
        thrs = thrs[~np.isnan(f1s)]
        f1s = f1s[~np.isnan(f1s)]
        best_thr = thrs[np.argmax(f1s)]
        return auc, f1, acc, prec, rec, best_thr, loss
    else:
        return auc, f1, acc, prec, rec, loss


def evaluate_ranking(model, eval_loader, device, eval_neg_num, evaluator):
    """
    evaluation for ranking
    :param model:
    :param eval_loader:
    :param device:
    :param eval_neg_num:
    :param evaluator:
    :return:
    """
    model.eval()
    pos_pred = []
    neg_pred = []
    eval_user = []
    eval_item = []
    with torch.no_grad():
        for user, item in eval_loader:
            output = model(user.view(-1).to(device), item.view(-1).to(device), eval_neg_num).view(-1, eval_neg_num+1)
            pos_pred.extend(list(output.data.cpu().numpy()[:, 0]))
            neg_pred.extend(list(output.data.cpu().numpy()[:, 1:]))
            eval_user.extend(user)
            eval_item.extend(item)

        performance = evaluator.evaluate_ranking(eval_user, eval_item, pos_pred, neg_pred)
        precision = performance[0]
        recall = performance[1]
        hit = performance[2]
        ndcg = performance[3]
    return precision, recall, hit, ndcg


def collate_fn(batch):
    """
    sample = {
            'user': np.array([user]),
            'item': np.array([item]),
            'label': np.array([label]),
            'act': np.array(act, dtype=object),
        }
    :param batch:
    :return:
    """
    users = [x['user'] for x in batch]
    users = torch.Tensor(users)
    items = [x['item'] for x in batch]
    items = torch.Tensor(items)
    labels = [x['label'] for x in batch]
    labels = torch.Tensor(labels)
    acts = [item['act'] for item in batch]
    return {
            'user': users,
            'item': items,
            'label': labels,
            'act': acts,
        }


def main(args):
    """
    main function of the model
    :param args: the args for model setting
    :return:
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
    t0 = time.time()
    log.record('Loading data...')
    train_data_helper = DataUtil(args, root_dir, 'train')
    valid_data_helper = DataUtil(args, root_dir, 'val')
    test_data_helper = DataUtil(args, root_dir, 'test')
    # eval_data_helper = DataUtil(args, root_dir, 'eval')

    train_loader = torch.utils.data.DataLoader(train_data_helper, batch_size=args.batch_size,
                                               shuffle=True, collate_fn=collate_fn)
    valid_loader = torch.utils.data.DataLoader(valid_data_helper, batch_size=args.test_batch_size,
                                               shuffle=True, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_data_helper, batch_size=args.test_batch_size,
                                              shuffle=True, collate_fn=collate_fn)

    # eval_data = eval_data_helper.get_data_4ranking()
    # eval_set = torch.utils.data.TensorDataset(torch.LongTensor(eval_data[0]), torch.LongTensor(eval_data[1]))
    # eval_loader = torch.utils.data.DataLoader(eval_set, batch_size=args.test_batch_size, shuffle=True)

    user_profile = torch.from_numpy(np.load(os.path.join(root_dir+'/data/'+args.dataset_name, args.user_profile))).float().to(device)
    item_profile = torch.from_numpy(np.load(os.path.join(root_dir+'/data/'+args.dataset_name, args.item_profile))).float().to(device)
    if args.dataset_name == 'yelp':
        biz_profile = None
        args.biz_num = 0
        u_bizs, i_bizs = {}, {}
    elif args.dataset_name == 'wxt':
        biz_profile = torch.from_numpy(np.load(os.path.join(root_dir+'/data/'+args.dataset_name, args.biz_profile))).float().to(device)
        u_bizs, i_bizs = train_data_helper.load_biz_data()

    social_relation = train_data_helper.load_social_data()
    t1 = time.time()
    log.record('--- # interaction links: {}, social links: {}, #u_bizs: {}, i_bizs:{}'.
               format(train_data_helper.data_size+valid_data_helper.data_size+test_data_helper.data_size,
                      sum(map(lambda x: len(x), social_relation.values()))/2, len(u_bizs), len(i_bizs)))
    log.record('--- #train size:{} #user:{}, #item:{}'.format(train_data_helper.data_size,
                                                              len(train_data_helper.u_items),
                                                              len(train_data_helper.i_users)))
    log.record('--- #valid size:{} #user:{}, #item:{}'.format(valid_data_helper.data_size,
                                                              len(valid_data_helper.u_items),
                                                              len(valid_data_helper.i_users)))
    log.record('--- #test size:{} #user:{}, #item:{}'.format(test_data_helper.data_size,
                                                             len(test_data_helper.u_items),
                                                             len(test_data_helper.i_users)))
    log.record('Load data finished {}s'.format(t1-t0))

    log.record('Initializing model...')
    heteinf = HeteInf(data_set=args.dataset_name, user_num=args.user_num, item_num=args.item_num,  biz_num=args.biz_num,
                      user_profile=user_profile, item_profile=item_profile, biz_profile=biz_profile,
                      user_items=train_data_helper.u_items, item_users=train_data_helper.i_users,
                      social_rel=social_relation, user_bizs=u_bizs, item_bizs=i_bizs,
                      emb_size=args.emb_size, profile_size=args.profile_size,
                      device=device).to(device)
    optimizer = torch.optim.Adam(heteinf.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # heteinf = nn.DataParallel(heteinf)

    log.record('Training model...')
    evaluator = Evaluation(args)

    # training and test
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, _, _ = train(heteinf, train_loader, valid_loader, test_loader, optimizer, device)
        t1 = time.time()
        # validation
        auc, f1, acc, pre, rec, best_thr, valid_loss = evaluate_clf(heteinf, valid_loader, device, return_best_thr=True)

        # classification on test set
        auc, f1, acc, pre, rec, test_loss = evaluate_clf(heteinf, test_loader, device, thr=best_thr)  # TODO: yelp's thr=0.5!!
        t2 = time.time()

        log.record('Epoch:%d, time: %.4fs, train loss:%.4f, val loss:%.4f, test loss:%.4f' %
                   (epoch, (t1 - t0), train_loss, valid_loss, test_loss))
        log.record("Classification, time:%.4fs, best threshold:%.4f ,"
                   "auc: %.4f, f1: %.4f, acc: %.4f, pre: %.4f, rec: %.4f" % ((t2 - t1), best_thr, auc, f1, acc, pre, rec))
        # # ranking on test set
        # precision, recall, hit, ndcg = evaluate_ranking(heteinf, eval_loader, device, args.eval_num, evaluator)
        # t3 = time.time()
        # log.record("Ranking, time:%.4fs , precision@k: %.4f, recall@k:%.4f, hit@k: %.4f, ndcg@k:%.4f" %
        #            ((t3 - t2), precision, recall, hit, ndcg))
        if args.save_model and epoch % 5 == 0:
            torch.save(heteinf.state_dict(), os.path.join(log_dir, args.dataset_name+'.'+str(epoch)+'.'
                                                          +str(round(auc,5))+'.'+str(round(best_thr,5))+'.model'))


def att_analysis(args, optimized_model_file):
    """
    attention analysis
    :param args:
    :param optimized_model_file:
    :return:
    """
    log.record('attention analysis...')
    log.record('loading data...')
    all_data_helper = DataUtil(args, root_dir, 'att_analysis')

    all_data_loader = torch.utils.data.DataLoader(all_data_helper, batch_size=args.batch_size,
                                                  shuffle=True, collate_fn=collate_fn)
    user_profile = torch.from_numpy(
        np.load(os.path.join(root_dir + '/data/' + args.dataset_name, args.user_profile))).float().to(device)
    item_profile = torch.from_numpy(
        np.load(os.path.join(root_dir + '/data/' + args.dataset_name, args.item_profile))).float().to(device)
    if args.dataset_name == 'yelp':
        biz_profile = None
        args.biz_num = 0
        u_bizs, i_bizs = {}, {}
    elif args.dataset_name == 'wxt':
        biz_profile = torch.from_numpy(
            np.load(os.path.join(root_dir + '/data/' + args.dataset_name, args.biz_profile))).float().to(device)
        u_bizs, i_bizs = all_data_helper.load_biz_data()

    social_relation = all_data_helper.load_social_data()
    log.record('--- #data size:{} #user:{}, #item:{}'.format(all_data_helper.data_size,
                                                              len(all_data_helper.u_items),
                                                              len(all_data_helper.i_users)))
    heteinf = HeteInf(data_set=args.dataset_name, user_num=args.user_num, item_num=args.item_num, biz_num=args.biz_num,
                      user_profile=user_profile, item_profile=item_profile, biz_profile=biz_profile,
                      user_items=all_data_helper.u_items, item_users=all_data_helper.i_users,
                      social_rel=social_relation, user_bizs=u_bizs, item_bizs=i_bizs,
                      emb_size=args.emb_size, profile_size=args.profile_size,
                      device=device).to(device)

    heteinf.eval()
    log.record('loading model...')
    heteinf.load_state_dict(torch.load(optimized_model_file))

    log.record('evaluation...')
    loss = 0.0
    criterion = nn.BCELoss()
    for i, data in tqdm(enumerate(all_data_loader, 0)):
        batch_user, batch_item, batch_label, batch_act = data['user'].long().to(device), \
                                                         data['item'].long().to(device), \
                                                         data['label'].float().to(device), \
                                                         data['act']
        output = heteinf(batch_user, batch_item, batch_act)
        loss += criterion(output, batch_label).data

    log.record('all data loss:{}'.format(loss))
    log.record('save attention...')
    np.save(optimized_model_file+'.inf_attention', np.array(heteinf.inf_att_analysis))
    np.save(optimized_model_file+'.item_attention', np.array(heteinf.item_fea_att_analysis))
    np.save(optimized_model_file+'.user_attention', np.array(heteinf.user_fea_att_analysis))


if __name__ == "__main__":
    time_stamp = datetime.datetime.now()
    print(time_stamp.strftime('%Y.%m.%d-%H:%M:%S'))

    parser = argparse.ArgumentParser(description='Social Influence Prediction over Heterogeneous Graph')
    parser.add_argument('--on_YARD', type=bool, default=False, help='run code on YARD')
    parser.add_argument('--save_model', type=bool, default=True, help='save model')
    # # # #
    parser.add_argument('--dataset_name', type=str, default='yelp', help='dataset name')
    parser.add_argument('--user_num', type=int, default=8163, help='the number of user')
    parser.add_argument('--item_num', type=int, default=7900, help='the number of item')
    parser.add_argument('--profile_size', type=int, default=150, metavar='N', help='profile size')

    # parser.add_argument('--dataset_name', type=str, default='wxt', help='dataset name')
    # parser.add_argument('--user_num', type=int, default=72371, help='the number of user')
    # parser.add_argument('--item_num', type=int, default=22218, help='the number of item')
    # parser.add_argument('--biz_num', type=int, default=218887, help='the number of bizs')
    # parser.add_argument('--profile_size', type=int, default=128, metavar='N', help='profile size')

    parser.add_argument('--user_profile', type=str, default='user_profile.npy', help='user profile name')
    parser.add_argument('--item_profile', type=str, default='item_profile.npy', help='item profile name')
    parser.add_argument('--biz_profile', type=str, default='biz_profile.npy', help='item profile name')
    parser.add_argument('--batch_size', type=int, default=1024, metavar='N', help='input batch size for training')
    parser.add_argument('--emb_size', type=int, default=64, metavar='N', help='embedding size')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay (L2 loss on parameters).')
    parser.add_argument('--test_batch_size', type=int, default=1024, metavar='N', help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
    parser.add_argument('--eval_num', type=int, default=100, help='the number of evaluation sample for ranking')
    parser.add_argument('--top_k', type=int, default=10, help='top-k for evaluation ranking performance')
    parser.add_argument('--worker_num', type=int, default=0, help='the number of worker for sampling')
    parser.add_argument('--seed', type=int, default=72, help='Random seed.')
    parser.add_argument('--use_cuda', type=bool, default=False, metavar='U', help='use cuda')
    parser.add_argument('--inf_flag', type=bool, default=True, help='Add social influence propagation.')
    args = parser.parse_args()

    # training settings
    if args.on_YARD:
        root_dir = '/mnt/yardcephfs/mmyard/g_wxg_sd_search/lucasyflu/HeteInf/'
    else:
        root_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))  # '..../HeteInf/', absolute dir

    # log settings
    log_dir = os.path.join(root_dir + '/code/', 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # define log name
    log_path = os.path.join(root_dir + '/code/', 'log/%s_%s.log' % (args.dataset_name, str(int(time.time()))))
    log = Logging(log_path)
    log.record(args)

    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    use_cuda = args.use_cuda
    if torch.cuda.is_available():
        log.record('Using cuda...')
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    main(args)

    # # # saved_model_file = 'log/yelp.10.0.9627152757200157.model'
    # # # saved_model_file = 'log/wxt.5.0.6903097156219249.0.1964704990386963.model'
    # saved_model_file = 'log/wxt.5.0.6681.0.19198.model'
    # att_analysis(args, saved_model_file)

