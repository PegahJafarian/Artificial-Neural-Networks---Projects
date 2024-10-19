import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
from interactions import Interactions
from eval_metrics import *
import argparse
import logging
import datetime
from model.KERL import kerl
import random
import pickle

class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path, change_opt=True):
        print(path)
        data = torch.load(path)
        if 'opt' in data:
            if change_opt:
                self.opt.parse(data['opt'], print_=False)
            self.load_state_dict(data['d'])
        else:
            self.load_state_dict(data)
        return self.cuda()

    def save(self, name=None,new=False):
        prefix = 'checkpoints/' + self.model_name + '_' +self.opt.type_+'_'
        if name is None:
            name = time.strftime('%m%d_%H:%M:%S.pth')
        path = prefix + name

        if new:
            data = {'opt':self.opt.state_dict(), 'd':self.state_dict()}
        else:
            data=self.state_dict()

        torch.save(data, path)
        return path

    def get_optimizer(self, lr1, lr2=0, weight_decay=0):
        ignored_params = list(map(id, self.embed.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, self.parameters())
        if lr2 is None: lr2 = lr1*0.5
        optimizer = torch.optim.Adam([
            dict(params=base_params,
                 weight_decay=weight_decay,
                 lr=lr1),
            {'params': self.embed.parameters(), 'lr':lr2}
        ])
        return optimizer



if __name__ == "__main__":
    print(1)
#----------------------------------------------------------------
from model.BasicModule import *
class DynamicGRU(BasicModule):
    def __init__(self, input_dim, output_dim,
                 num_layers=1, bidirectional=False,
                 batch_first=True):
        super().__init__()
        self.embed_dim = input_dim
        self.hidden_dim = output_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.gru = nn.GRU(self.embed_dim,
                            self.hidden_dim,
                            num_layers=self.num_layers,
                            bidirectional=self.bidirectional,
                            batch_first=self.batch_first)

    def forward(self, inputs, lengths,one = False):
 
        if one == True:
            hidden = lengths
            out, ht = self.gru(inputs,hidden)
        else:
            
            _, idx_sort = torch.sort(lengths, dim=0, descending=True)
            _, idx_unsort = torch.sort(idx_sort, dim=0)
            sort_embed_input = inputs.index_select(0, Variable(idx_sort))
            sort_lengths = list(lengths[idx_sort])

           
            inputs_packed = nn.utils.rnn.pack_padded_sequence(sort_embed_input,
                                                              sort_lengths,
                                                              batch_first=True)
       
            out_pack, ht = self.gru(inputs_packed)

        
            out = nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=self.batch_first)
            out = out[0]

        
            ht = torch.transpose(ht, 0, 1)[idx_unsort]
            ht = torch.transpose(ht, 0, 1)
            out = out[idx_unsort]

        return out, ht
    
#--------------------------------------

import torch.nn as nn

from eval_metrics import *
from model.DynamicGRU import DynamicGRU
class kerl(nn.Module):
    def __init__(self, num_users, num_items, model_args, device,kg_map):
        super(kerl, self).__init__()

        self.args = model_args
        self.device = device
        self.lamda = 10
        # init args
        L = self.args.L
        dims = self.args.d
        predict_T=self.args.T
        # user and item embeddings
        self.kg_map =kg_map

        self.item_embeddings = nn.Embedding(num_items, dims).to(device)
        self.DP = nn.Dropout(0.5)
        self.enc = DynamicGRU(input_dim=dims,
                              output_dim=dims, bidirectional=False, batch_first=True)


        self.mlp = nn.Linear(dims+50*2, dims*2)
        self.fc = nn.Linear(dims*2, num_items)
        self.mlp_history = nn.Linear(50,50)
        self.BN = nn.BatchNorm1d(50, affine=False)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, batch_sequences, train_len):
      
        probs = []
        input = self.item_embeddings(batch_sequences)
        out_enc, h = self.enc(input, train_len)

        kg_map = self.BN(self.kg_map)
        kg_map =kg_map.detach()
        batch_kg = self.get_kg(batch_sequences,train_len,kg_map)

        mlp_in = torch.cat([h.squeeze(),batch_kg, self.mlp_history(batch_kg)],dim=1)
        mlp_hidden = self.mlp(mlp_in)
        mlp_hidden = torch.tanh(mlp_hidden)

        out = self.fc(mlp_hidden)
        probs.append(out)
        return torch.stack(probs, dim=1)

    def RLtrain(self, batch_sequences, items_to_predict, pred_one_hot, train_len,tarlen):
        probs = []
        probs_orgin = []
        each_sample = [] 
        Rewards = []
        input = self.item_embeddings(batch_sequences)

        out_enc, h = self.enc(input, train_len)

        kg_map = self.BN(self.kg_map)
        batch_kg = self.get_kg(batch_sequences,train_len,kg_map)
       
        mlp_in = torch.cat([h.squeeze(),batch_kg,self.mlp_history(batch_kg)],dim=1)
        mlp_hidden = self.mlp(mlp_in)
        mlp_hidden = torch.tanh(mlp_hidden)
        out_fc = self.fc(mlp_hidden)

        out_distribution = F.softmax(out_fc, dim=1)
        probs_orgin.append(out_distribution)
        out_distribution = 0.8 * out_distribution
        out_distribution = torch.add(out_distribution, pred_one_hot)
   
        probs.append(out_distribution)
        m = torch.distributions.categorical.Categorical(out_distribution)
    
        sample1 = m.sample()
        each_sample.append(sample1)
      
        Reward, dist_sort = self.generateReward(sample1, self.args.T-1, 3, items_to_predict, pred_one_hot, h,batch_kg,kg_map,tarlen)
        Rewards.append(Reward)
       

        probs = torch.stack(probs, dim=1)
        probs_orgin = torch.stack(probs_orgin, dim=1)
        return probs, probs_orgin, torch.stack(each_sample, dim=1), torch.stack(Rewards, dim=1),dist_sort

    def get_kg(self,batch_sequences,trainlen,kg_map):
    
        batch_kg = []
        for i, seq in enumerate(batch_sequences):
            seq_kg = kg_map[seq]
            seq_kg_avg = torch.sum(seq_kg,dim=0)
            seq_kg_avg = torch.div(seq_kg_avg,trainlen[i])
            batch_kg.append(seq_kg_avg)
        batch_kg = torch.stack(batch_kg)
        return batch_kg

    def generateReward(self, sample1, path_len, path_num, items_to_predict, pred_one_hot,h_orin,batch_kg,kg_map,tarlen):
        history_kg = self.mlp_history(batch_kg)
        Reward = []
        dist = []
        dist_replay = []
        for paths in range(path_num):
            h = h_orin
            indexes = []
            indexes.append(sample1)
            dec_inp_index = sample1
            dec_inp = self.item_embeddings(dec_inp_index)
            dec_inp = dec_inp.unsqueeze(1)
            ground_kg = self.get_kg(items_to_predict[:, self.args.T - path_len - 1:],tarlen,kg_map)
            for i in range(path_len):
                out_enc, h = self.enc(dec_inp, h, one=True)
                
                mlp_in = torch.cat([h.squeeze(), batch_kg, self.mlp_history(batch_kg)], dim=1)
                mlp_hidden = self.mlp(mlp_in)
                mlp_hidden = torch.tanh(mlp_hidden)
                out_fc = self.fc(mlp_hidden)

                out_distribution = F.softmax(out_fc, dim=1)
                out_distribution = 0.8 * out_distribution
                out_distribution = torch.add(out_distribution, pred_one_hot)
            
                m = torch.distributions.categorical.Categorical(out_distribution)
                sample2 = m.sample()
                dec_inp = self.item_embeddings(sample2)
                dec_inp = dec_inp.unsqueeze(1)
                indexes.append(sample2)
            indexes = torch.stack(indexes, dim=1)
            episode_kg = self.get_kg(indexes,torch.Tensor([path_len+1]*len(indexes)),kg_map)


            dist.append(self.cos(episode_kg ,ground_kg))
            dist_replay.append(self.cos(episode_kg,history_kg))
           
            Reward.append(dcg_k(items_to_predict[:, self.args.T - path_len - 1:], indexes, path_len + 1))
        Reward = torch.FloatTensor(Reward).to(self.device)
        dist = torch.stack(dist, dim=0)
        dist = torch.mean(dist, dim=0)

        dist_replay = torch.stack(dist_replay, dim=0)
        dist_sort = self.compare_kgReawrd(Reward, dist_replay)
        Reward = torch.mean(Reward, dim=0)
        Reward = Reward + self.lamda * dist
        dist.sort =dist_sort.detach()
        return Reward, dist_sort


    def compare_kgReawrd(self, reward, dist):
        logit_reward, indice = reward.sort(dim=0)
        dist_sort = dist.gather(dim=0, index=indice)
        return dist_sort
    
#----------------------------------------------------------

def bleu(hyps, refs):
 
    bleu_4 = []
    for hyp, ref in zip(hyps, refs):
        try:
            score = bleu_score.sentence_bleu(
                [ref], hyp,
                smoothing_function=SmoothingFunction().method7,
                weights=[0.5, 0.5, 0, 0])
        except:
            score = 0
        bleu_4.append(score)
    bleu_4 = np.average(bleu_4)
    return bleu_4

def bleu_each(hyps, refs):
 
    bleu_4 = []
    hyps=hyps.cpu().numpy()
    refs=refs.cpu().numpy()
    for hyp, ref in zip(hyps, refs):
        try:
            score = bleu_score.sentence_bleu(
                [ref], hyp,
                smoothing_function=SmoothingFunction().method7,
                weights=[0.5, 0.5, 0, 0])
        except:
            score = 0
        bleu_4.append(score)
    return bleu_4



def precision_at_k(actual, predicted, topk,item_i):
    sum_precision = 0.0
    user = 0
    num_users = len(predicted)
    for i in range(num_users):
        if actual[i][item_i]>0:
            user +=1
            act_set = actual[i][item_i]
            pred_set = predicted[i]
            if act_set in pred_set:
                sum_precision += 1
        else:
            continue
  
    return sum_precision / user


def ndcg_k(actual, predicted, topk,item_i):
    k = min(topk, len(actual))
    idcg = idcg_k(k)
    res = 0
    user = 0
    for user_id in range(len(actual)):
        if actual[user_id][item_i] > 0:
            user +=1
            dcg_k = sum([int(predicted[user_id][j] in [actual[user_id][item_i]]) / math.log(j+2, 2) for j in range(k)])
            res += dcg_k
        else:
            continue

    return res/user

def dcg_k(actual, predicted, topk):
    k = min(topk, len(actual))
    dcgs=[]
    actual = actual.cpu().numpy()
    predicted = predicted.cpu().numpy()
    for user_id in range(len(actual)):
        value = []
        for i in predicted[user_id]:
            try:
                value += [topk -int(np.argwhere(actual[user_id]==i))]
      
            except:
                value += [0]
      
        dcg_k = sum([value[j] / math.log(j+2, 2) for j in range(k)])
        if dcg_k==0:
           dcg_k=1e-5
        dcgs.append(dcg_k)
    return dcgs

def idcg_k(k):
    res = sum([1.0/math.log(i+2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res
    
#-----------------------------------------------------

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

def generate_testsample(test_set,itemnum):


    all_sample =[]
    for eachset in test_set:
        testsample = []
        for i in range(1):
            onesample = []
            onesample +=[eachset[i]]
            other = list(range(1, itemnum))
            other.remove(eachset[i])
            neg = random.sample(other,100)
            onesample +=neg
            testsample.append(onesample)
        testsample = np.stack(testsample)
        all_sample.append(testsample)
    all_sample = np.stack(all_sample)
    return all_sample

def evaluation_kerl(kerl, train, test_set):
    num_users = train.num_users
    num_items = train.num_items
    batch_size = 1024
    num_batches = int(num_users / batch_size) + 1
    user_indexes = np.arange(num_users)
    item_indexes = np.arange(num_items)
    pred_list = None
    test_sequences = train.test_sequences.sequences
    test_len = train.test_sequences.length

    all_sample = generate_testsample(test_set,num_items)
    for batchID in range(num_batches):
        start = batchID * batch_size
        end = start + batch_size

        if batchID == num_batches - 1:
            if start < num_users:
                end = num_users
            else:
                break

        batch_user_index = user_indexes[start:end]

        batch_test_sequences = test_sequences[batch_user_index]
        batch_test_sequences = np.atleast_2d(batch_test_sequences)
        batch_test_len = test_len[batch_user_index]

        batch_test_len = torch.from_numpy(batch_test_len).type(torch.LongTensor).to(device)
        batch_test_sequences = torch.from_numpy(batch_test_sequences).type(torch.LongTensor).to(device)

        prediction_score = kerl(batch_test_sequences, batch_test_len)
        rating_pred = prediction_score
        rating_pred = rating_pred.cpu().data.numpy().copy()

        if batchID == 0:
            pred_list = rating_pred
        else:
            pred_list = np.append(pred_list, rating_pred, axis=0)

   
    all_top10 = []
    for i in range(1):
        oneloc_top10 = []
        user_index = 0
        for each_policy,each_s in zip(pred_list[:, i, :],all_sample[:,i,:]):
        
            each_sample = -each_policy[each_s]
            top10index = np.argsort(each_sample)[:10]
            top10item = each_s[top10index]
            oneloc_top10.append(top10item)
        oneloc_top10=np.stack(oneloc_top10)
        all_top10.append(oneloc_top10)
        user_index +=1
    all_top10 = np.stack(all_top10,axis=1)
    pred_list = all_top10

    precision, ndcg = [], []
    k=10
    for i in range(1):
        pred = pred_list[:,i,:]
        precision.append(precision_at_k(test_set, pred, k,i))
        ndcg.append(ndcg_k(test_set, pred, k,i))

  
    return precision, ndcg

def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=100):
    # early stopping strategy:
    assert expected_order in ['acc', 'dec']

    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop

def train_kerl(train_data, test_data, config,kg_map):
    num_users = train_data.num_users
    num_items = train_data.num_items

 
    sequences_np = train_data.sequences.sequences
    targets_np = train_data.sequences.targets
    users_np = train_data.sequences.user_ids
    trainlen_np = train_data.sequences.length
    tarlen_np = train_data.sequences.tarlen

    n_train = sequences_np.shape[0]
    logger.info("Total training records:{}".format(n_train))


    kg_map = torch.from_numpy(kg_map).type(torch.FloatTensor).to(device)
    kg_map.requires_grad=False
    seq_model = kerl(num_users, num_items, config, device, kg_map).to(device)
    optimizer = torch.optim.Adam(seq_model.parameters(), lr=config.learning_rate,weight_decay=config.l2)

    lamda = 5  #loss function hyperparameter
    print("loss lamda=",lamda)
    CEloss = torch.nn.CrossEntropyLoss()
    margin = 0.0
    MRLoss = torch.nn.MarginRankingLoss(margin=margin)

    record_indexes = np.arange(n_train)
    batch_size = config.batch_size
    num_batches = int(n_train / batch_size) + 1

    stopping_step = 0
    cur_best_pre_0 = 0
    should_stop = False
    for epoch_num in range(config.n_iter):
        t1 = time()
        loss=0
    
        seq_model.train()

        np.random.shuffle(record_indexes)
        epoch_reward=0.0
        epoch_loss = 0.0
        for batchID in range(num_batches):
            start = batchID * batch_size
            end = start + batch_size

            if batchID == num_batches - 1:
                if start < n_train:
                    end = n_train
                else:
                    break

            batch_record_index = record_indexes[start:end]

            batch_users = users_np[batch_record_index]
            batch_sequences = sequences_np[batch_record_index]
            batch_targets = targets_np[batch_record_index]
            trainlen = trainlen_np[batch_record_index]
            tarlen = tarlen_np[batch_record_index]

            tarlen = torch.from_numpy(tarlen).type(torch.LongTensor).to(device)
            trainlen = torch.from_numpy(trainlen).type(torch.LongTensor).to(device)
            batch_users = torch.from_numpy(batch_users).type(torch.LongTensor).to(device)
            batch_sequences = torch.from_numpy(batch_sequences).type(torch.LongTensor).to(device)
            batch_targets = torch.from_numpy(batch_targets).type(torch.LongTensor).to(device)

            items_to_predict = batch_targets

            if epoch_num>=0:
                pred_one_hot = np.zeros((len(batch_users),num_items))
                batch_tar=targets_np[batch_record_index]
                for i,tar in enumerate(batch_tar):
                    pred_one_hot[i][tar]=0.2/config.T
                pred_one_hot = torch.from_numpy(pred_one_hot).type(torch.FloatTensor).to(device)

                prediction_score,orgin,batch_targets,Reward,dist_sort = seq_model.RLtrain(batch_sequences,
                items_to_predict,pred_one_hot,trainlen,tarlen)

                target = torch.ones((len(prediction_score))).to(device)

                min_reward = dist_sort[0,:].unsqueeze(1)
                max_reward = dist_sort[-1,:].unsqueeze(1)
                mrloss = MRLoss(max_reward,min_reward,target)

                orgin = orgin.view(prediction_score.shape[0] * prediction_score.shape[1], -1)
                target = batch_targets.view(batch_targets.shape[0]*batch_targets.shape[1])
                reward = Reward.view(Reward.shape[0]*Reward.shape[1]).to(device)

                prob = torch.index_select(orgin,1,target)
                prob = torch.diagonal(prob,0)
                RLloss =-torch.mean(torch.mul(reward,torch.log(prob)))
                loss = RLloss+lamda*mrloss
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        epoch_loss /= num_batches
        t2 = time()
        output_str = "Epoch %d [%.1f s]  loss=%.4f" % (epoch_num + 1, t2 - t1, epoch_loss)
        logger.info(output_str)

        if (epoch_num + 1) > 1:
            seq_model.eval()
            precision, ndcg = evaluation_kerl(seq_model, train_data, test_data)

            logger.info(', '.join(str(e) for e in precision))
            logger.info(', '.join(str(e) for e in ndcg))
            logger.info("Evaluation time:{}".format(time() - t2))
            cur_best_pre_0, stopping_step, should_stop = early_stopping(precision[0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc',
                                                                    flag_step=5)

        
            if should_stop == True:
                break
    logger.info("\n")
    logger.info("\n")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
 
    parser.add_argument('--L', type=int, default=50)
    parser.add_argument('--T', type=int, default=3)


    parser.add_argument('--n_iter', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--l2', type=float, default=0)

  
    parser.add_argument('--d', type=int, default=50)

    config = parser.parse_args()

    from data import Amazon
    data_set = Amazon.Beauty()  
    train_set, test_set, num_users, num_items,kg_map = data_set.generate_dataset(index_shift=1)

    maxlen = 0
    for inter in train_set:
        if len(inter)>maxlen:
            maxlen=len(inter)

    train = Interactions(train_set, num_users, num_items)
    train.to_newsequence(config.L, config.T)

    logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info(config)

    train_kerl(train,test_set,config,kg_map)

