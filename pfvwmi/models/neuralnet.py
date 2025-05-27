
from copy import deepcopy
import numpy as np
import os
import torch
from torch.nn.functional import relu





class FFNN(torch.nn.Module):

    def __init__(self, dimensions, discrete_output):
        super(FFNN, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dimensions = dimensions        
        self.discrete_output = discrete_output

        # build fully-connected feed-forward net
        seq = []
        for i in range(len(self.dimensions)-2):
            seq.append(torch.nn.Linear(self.dimensions[i], self.dimensions[i+1]))
            seq.append(torch.nn.ReLU())

        seq.append(torch.nn.Linear(self.dimensions[-2], self.dimensions[-1]))
        self.layers = torch.nn.Sequential(*seq)
        self.to(self.device)

    def evaluate(self, x):
        x = torch.Tensor(x).to(self.device)
        with torch.no_grad():
            logits = self(x)
            if self.discrete_output:
                if self.dimensions[-1] == 1:
                    return torch.where(torch.sigmoid(logits) > 0.5,
                                       torch.ones(logits.shape).to(self.device),
                                       torch.zeros(logits.shape).to(self.device))
                else:
                    return torch.argmax(logits, 1)
            else:
                return logits


    def save(self, path):
        d = {'sd' : deepcopy(self.state_dict()),
             'dimensions' : self.dimensions,
             'discrete_output' : self.discrete_output}
        torch.save(d, path)
        

    @staticmethod
    def load(path):
        d = torch.load(path, weights_only=True)
        net = FFNN(d['dimensions'],
                   d['discrete_output'])
        net.load_state_dict(d['sd'])
        return net

    def forward(self, x):
        return self.layers(x)

    def test_performance(self, data):
        out_dim = self.dimensions[-1]
        x = torch.Tensor(data[:, :-out_dim]).to(self.device)
        y_target = torch.Tensor(data[:, -out_dim:]).to(self.device)

        with torch.no_grad():
            logits = self(x)
            if self.discrete_output:
                if out_dim == 1:
                    y_pred = torch.where(torch.sigmoid(logits) > 0.5,
                                         torch.ones(logits.shape).to(self.device),
                                         torch.zeros(logits.shape).to(self.device))

                    metric = (y_pred == y_target).sum().cpu().item() / len(data)
                else:
                    metric = (torch.argmax(logits, 1) ==
                              torch.argmax(y_target, 1)).sum().cpu().item() / len(data)
                    
            else:
                y_pred = logits
                metric = ((y_pred - y_target)**2).sum().cpu().item() / len(data)

        return metric

    @staticmethod
    def train_FFNN(data, dimensions, discrete_output, epochs, model_path_f=None,
                   seed=666, batch_size=1000, lr=1e-3, momentum=0.9,
                   checkpoint_delta=None, valid_split=0.1):

        assert(dimensions[0] + dimensions[-1] == data.shape[1])

        np.random.seed(seed)
        torch.manual_seed(seed)

        if checkpoint_delta is None:
            checkpoint_delta = epochs // 10

        net = FFNN(dimensions, discrete_output)

        if net.discrete_output:
            if dimensions[-1] == 1:
                loss_fn = torch.nn.BCEWithLogitsLoss()
            else:
                loss_fn = torch.nn.CrossEntropyLoss()

            metric_str = "ACCURACY"

        else:
            loss_fn = torch.nn.MSELoss()
            metric_str = "MSE"

        optim = torch.optim.SGD(net.parameters(), lr=lr,
                                momentum=momentum)

        if model_path_f is not None:
            net.save(model_path_f(0))

        train_size = int(len(data) * (1 - valid_split))
        train, valid = data[:train_size], data[train_size:]
        best_metric, best_sd = None, None

        for epoch in range(epochs):

            np.random.shuffle(train)
            train_batches = [train[i:i+batch_size, ...]
                             for i in range(0, train.shape[0], batch_size)]

            tot_loss, train_metric = 0.0, 0.0
            net.train()
            
            for batchnum, batch in enumerate(train_batches):
                x = torch.Tensor(batch[:, :-net.dimensions[-1]]).to(net.device)
                y_target = torch.Tensor(batch[:, -net.dimensions[-1]:]).to(net.device)

                optim.zero_grad()
                logits = net(x)
                loss = loss_fn(logits, y_target)

                loss.backward()
                optim.step()

                if net.discrete_output:
                    if net.dimensions[-1] == 1:
                        y_pred = torch.where(torch.sigmoid(logits) > 0.5,
                                             torch.ones(logits.shape).to(net.device),
                                             torch.zeros(logits.shape).to(net.device))

                        metric = (y_pred == y_target).sum().cpu().item() / len(batch)
                    else:
                        metric = (torch.argmax(logits, 1) ==
                                  torch.argmax(y_target, 1)).sum().cpu().item() / len(batch)
                    
                else:
                    y_pred = logits
                    metric = ((y_pred - y_target)**2).sum().cpu().item() / len(batch)

                train_metric += metric
                tot_loss += loss.item()

            avg_train_metric = train_metric / len(train_batches)
            avg_loss = tot_loss / len(train_batches)

            valid_batches = [valid[i:i+batch_size, ...]
                             for i in range(0, valid.shape[0], batch_size)]
            valid_metric = 0.0
            net.eval()
            for _, batch in enumerate(valid_batches):
                valid_metric += net.test_performance(batch)
    
            avg_valid_metric = valid_metric / len(valid_batches)

            if best_metric is None or avg_valid_metric > best_metric:
                best_metric = avg_valid_metric
                best_sd = deepcopy(net.state_dict())

            msg = f'EPOCH: {epoch}, LOSS: {avg_loss}'
            msg += f', TRAIN {metric_str}: {avg_train_metric}'
            msg += f', VALID {metric_str}: {avg_valid_metric}'
            print(msg)

            if model_path_f is not None \
               and ((epoch+1) % checkpoint_delta == 0):
                net.save(model_path_f(epoch+1))

        net.load_state_dict(best_sd)
        return net





if __name__ == '__main__':

    
    hidden = [2, 2]

    bp = [(1, 2), (-1, 0)]

    print("Bounds:")
    print("x1:", bp[0])
    print("x2:", bp[1])

    bounds = And(LE(Real(bp[0][0]), input_vars[0]),
                 LE(input_vars[0], Real(bp[0][1])),
                 LE(Real(bp[1][0]), input_vars[1]),
                 LE(input_vars[1], Real(bp[1][1])))

    dimensions = [2] + hidden + [1]
    discrete_output = True
    net = FFNN(dimensions, discrete_output)
    sd = net.state_dict()

    sd['layers.0.weight'] = torch.Tensor(np.array([1., 0., 0., 1.]).reshape((2,2)))
    sd['layers.0.bias'] = torch.Tensor([0., 0.])
    
    sd['layers.2.weight'] = torch.Tensor(np.array([1., 1., 1, -1.]).reshape((2,2)))
    sd['layers.2.bias'] = torch.Tensor([0., 0.])
    
    sd['layers.4.weight'] = torch.Tensor([[1., 1.]])
    sd['layers.4.bias'] = torch.Tensor([0.])
    
    net.load_state_dict(sd)

    
    print()
    print("--------------------")
    print("Encoding w/out boundprop")
    enc1 = net.smt_formula(input_vars, output_vars)
    for i, clause in enumerate(enc1.args()):
        print(f"clause {i}: {serialize(clause)}")

    if WMISolver is not None:
        solver = WMISolver(enc1, Real(1))
        wmi1, nint1 = solver.computeWMI(bounds, input_vars)
        print("encoding 1", wmi1, nint1)        

    print()
    print("--------------------")
    print("Encoding w boundprop")
    enc2 = net.smt_formula(input_vars, output_vars, bounds=bp)
    for i, clause in enumerate(enc2.args()):
        print(f"clause {i}: {serialize(clause)}")

    print()
    if WMISolver is not None:
        solver = WMISolver(enc2, Real(1))
        wmi2, nint2 = solver.computeWMI(bounds, input_vars)
        print("encoding 2", wmi2, nint2)


    
    

    
    

