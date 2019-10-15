import torch
import torch.nn as nn
from .optimizer import BertAdam as Adam
from .utils import AverageMeter


class DropBertModel():
    def __init__(self, args, network, state_dict=None, num_train_step=-1):
        self.args = args
        self.train_loss = AverageMeter()
        self.step = 0
        self.updates = 0
        self.network = network
        if state_dict is not None:
            print("Load Model!")
            self.network.load_state_dict(state_dict["state"])
        self.mnetwork = nn.DataParallel(self.network) if args.gpu_num > 1 else self.network

        self.total_param = sum([p.nelement() for p in self.network.parameters() if p.requires_grad])
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [p for n, p in self.network.bert.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.bert_weight_decay, 'lr': args.bert_learning_rate},
            {'params': [p for n, p in self.network.bert.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': args.bert_learning_rate},
            {'params': [p for n, p in self.network.named_parameters() if not n.startswith("bert.")],
             "weight_decay": args.weight_decay, "lr": args.learning_rate}
        ]
        self.optimizer = Adam(optimizer_parameters,
                              lr=args.learning_rate,
                              warmup=args.warmup,
                              t_total=num_train_step,
                              max_grad_norm=args.grad_clipping,
                              schedule=args.warmup_schedule)
        if self.args.gpu_num > 0:
            self.network.cuda()
        self.em_avg = AverageMeter()
        self.f1_avg = AverageMeter()

    def avg_reset(self):
        self.train_loss.reset()
        self.em_avg.reset()
        self.f1_avg.reset()

    def update(self, tasks):
        self.network.train()
        output_dict = self.mnetwork(**tasks)
        loss = output_dict["loss"]
        metrics = self.mnetwork.get_metrics(True)
        self.em_avg.update(metrics["em"], 1)
        self.f1_avg.update(metrics["f1"], 1)
        self.train_loss.update(loss.item(), 1)
        if self.args.gradient_accumulation_steps > 1:
            loss /= self.args.gradient_accumulation_steps
        loss.backward()
        if (self.step + 1) % self.args.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.updates += 1
        self.step += 1

    @torch.no_grad()
    def evaluate(self, dev_data_list):
        dev_data_list.reset()
        self.network.eval()
        loss_sum = 0
        total_batch = 0
        total_num = 0
        for batch in dev_data_list:
            total_num += batch["input_ids"].size(0)
            output_dict = self.network(**batch)
            loss_sum += output_dict["loss"].item()
            total_batch += 1
        metrics = self.network.get_metrics(True)
        self.network.train()

        return total_num, loss_sum / total_batch, metrics["em"], metrics["f1"]

    def save(self, prefix, epoch):
        network_state = dict([(k, v.cpu()) for k, v in self.network.state_dict().items()])
        other_params = {
            'optimizer': self.optimizer.state_dict(),
            'config': self.args,
            'epoch': epoch
        }
        state_path = prefix + ".pt"
        other_path = prefix + ".ot"
        torch.save(other_params, other_path)
        torch.save(network_state, state_path)
        print('model saved to {}'.format(prefix))