
import numpy as np
import pickle

from pfvwmi.models import DET


class DETForest:

    EPSILON = 1e-5    

    class DummyDET:

        def evaluate(self, x):
            return np.ones(shape=x.shape) * DETForest.EPSILON

        def smt_weight(out_var, bounds=None):
            from pysmt.shortcuts import Equals, Real
            return Equals(out_var, Real(EPSILON))

    @property
    def size(self):
        return [[[c[0].size, c[1].size] for c in out_comp]
                for out_comp in self.forest]

    def __init__(self, feats, train, forest_size, output_size,
                 n_min=None, n_max=None, seed=None):

        self.input_size = len(feats)
        self.output_size = output_size
        self.discrete_output = True

        if seed is not None:
            np.random.seed(seed)

        self.forest = []
        for yj in range(output_size):
            
            train_yj = [train[train[:,self.input_size + yj] == b][:, :self.input_size]
                        for b in [0,1]]
            size_yj = [len(train_yj[b]) for b in [0,1]]

            self.forest.append([])
            for i in range(forest_size):
                dets_yj_i = []
                for b in [0,1]:
                    bag_yj_i = train_yj[b][np.random.choice(range(size_yj[b]),
                                                            size_yj[b] // forest_size,
                                                            replace=False)]
                
                    if len(bag_yj_i) > 1:
                        adj_n_min = int(n_min * len(bag_yj_i)/len(train))
                        adj_n_max = int(n_max * len(bag_yj_i)/len(train))
                        dets_yj_i.append(DET(feats, bag_yj_i, adj_n_min, adj_n_max))

                    else:
                        print(f"Dummy DET {yj} {i}")
                        dets_yj_i.append(DETForest.DummyDET())

                self.forest[-1].append(tuple(dets_yj_i))

    def evaluate(self, x):
        y = np.array([[[c[0].evaluate(x), c[1].evaluate(x)] for c in out_comp]
                      for out_comp in self.forest]).sum(axis=1)
        labels = y.argmax(axis=1)
        #print("y", y)
        #print("labels", labels)
        return labels.T

    def test_performance(self, data):
        x = data[:, :-self.output_size]
        y_target = data[:, -self.output_size:]
        y_pred = self.evaluate(x)
        #print("ypred", y_pred)
        #print("ytarget", y_target)
        #print("(y_pred == y_target)", (y_pred == y_target))
        #print("(y_pred == y_target).shape", (y_pred == y_target).shape)
        #print("np.all(y_pred == y_target, axis=1)", np.all(y_pred == y_target, axis=1))
        #print("np.all(y_pred == y_target, axis=1).shape", np.all(y_pred == y_target, axis=1).shape)
        return np.mean(np.all(y_pred == y_target, axis=1))


    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)




if __name__ == '__main__':

    from test import generate_problem

    seed = 1337
    input_size = 4
    output_size = 1
    train = generate_problem(input_size, output_size, True)
    valid_test = generate_problem(input_size, output_size, True)
    valid = valid_test[:len(valid_test)//2]
    test = valid_test[len(valid_test)//2:]

    feats = [(f'x_{i}', 'real') for i in range(input_size)]

    forest_size = 20
    n_min = len(train) // 10
    n_max = len(train) // 2
    max_iter = 100

    print(f"Training ensemble of {forest_size} DET({n_min},{n_max})")
    ensemble = DETForest(feats, train, forest_size, output_size, n_min=n_min, n_max=n_max, seed=seed)    
    print("Accuracy:", ensemble.test_performance(test))
    print("Size:", ensemble.size)
