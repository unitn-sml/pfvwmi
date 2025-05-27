
import matplotlib.pyplot as plt
import numpy as np
import os

from pfvwmi.models.dataset import Dataset

# parameters

p_female = 0.5

hpd_mean = 7
hpd_std = 1.0

hpd_mean_biased = 7
hpd_std_biased = 1.0

min_hours = 2
max_hours = 12

yexp_mean = 20
yexp_std = 7
min_exp = 0
max_exp = 55

threshold = 35e3

workdays = 5
workweeks = 48

k_mean = 1
k_std = 0.02

gap_mean = 0.1
gap_std = 0.02


def sample(n, unbiased=True):
    
    female = (np.random.random(n) < p_female).astype(int)

    if unbiased:
        hpd = (female * np.random.normal(hpd_mean, hpd_std, n)) +\
            ((1 - female) * np.random.normal(hpd_mean, hpd_std, n))
    else:
        hpd = (female * np.random.normal(hpd_mean_biased, hpd_std_biased, n)) +\
            ((1 - female) * np.random.normal(hpd_mean, hpd_std, n))
        
    hpd = np.minimum(hpd, max_hours*np.ones(n))
    hpd = np.maximum(min_hours * np.ones(n), hpd)
    hpw = workdays * hpd

    yexp = np.random.normal(yexp_mean, yexp_std, n)
    yexp = np.minimum(yexp, max_exp * np.ones(n))
    yexp = np.maximum(min_exp * np.ones(n), yexp)

    age = yexp + 16 + np.minimum(np.zeros(n), np.random.normal(6, 10, n))

    hourly_wage = np.random.normal(k_mean, k_std, n) * (500 * yexp + 30e3) / (8 * workdays * workweeks)
    if not unbiased:
        hourly_wage  = hourly_wage  + hourly_wage * np.random.normal(gap_mean, gap_std, n) * (1 - 2 * female)

    income = (hpw * workweeks) * hourly_wage

    label = (income > threshold).astype(int)

    return np.concatenate((female.reshape(-1,1),
                           hpw.reshape(-1,1),
                           yexp.reshape(-1,1),
                           hourly_wage.reshape(-1,1),
                           income.reshape(-1,1),
                           age.reshape(-1,1),
                           label.reshape(-1,1)), axis=1)

def plot_statistics(X_unbiased, X_biased):

    assert(len(X_unbiased) == len(X_biased))
    n_samples = len(X_unbiased)

    pointsize = 1
    pointalpha = 0.15
    #figsize = (19.2, 10.8)

    unbiased_female, unbiased_male = [], []
    biased_female, biased_male = [], []
    for i in range(n_samples):
        if X_unbiased[i][0]: unbiased_female.append(X_unbiased[i][1:])
        else: unbiased_male.append(X_unbiased[i][1:])
        if X_biased[i][0]: biased_female.append(X_biased[i][1:])
        else: biased_male.append(X_biased[i][1:])

    unbiased_female, unbiased_male = np.array(unbiased_female), np.array(unbiased_male)
    biased_female, biased_male = np.array(biased_female), np.array(biased_male)

    fig, ax = plt.subplots(3,2)
    #fig.set_size_inches(*figsize)
    #fig.suptitle("Synthetic benchmark")

    ax[0,0].scatter(unbiased_male[:,0], unbiased_male[:,3],
                    color='c', alpha=pointalpha, s=pointsize)
    ax[0,0].scatter(unbiased_female[:,0], unbiased_female[:,3],
                    color='m', alpha=pointalpha, s=pointsize)

    ax[0,1].scatter(biased_male[:,0], biased_male[:,3],
                    color='c', alpha=pointalpha, s=pointsize)
    ax[0,1].scatter(biased_female[:,0], biased_female[:,3],
                    color='m', alpha=pointalpha, s=pointsize)

    ax[1,0].scatter(unbiased_male[:,1], unbiased_male[:,3],
                    color='c', alpha=pointalpha, s=pointsize)
    ax[1,0].scatter(unbiased_female[:,1], unbiased_female[:,3],
                    color='m', alpha=pointalpha, s=pointsize)

    ax[1,1].scatter(biased_male[:,1], biased_male[:,3],
                    color='c', alpha=pointalpha, s=pointsize)
    ax[1,1].scatter(biased_female[:,1], biased_female[:,3],
                    color='m', alpha=pointalpha, s=pointsize)

    ax[2,0].scatter(unbiased_male[:,1], unbiased_male[:,2],
                    color='c', alpha=pointalpha, s=pointsize)
    ax[2,0].scatter(unbiased_female[:,1], unbiased_female[:,2],
                    color='m', alpha=pointalpha, s=pointsize)

    ax[2,1].scatter(biased_male[:,1], biased_male[:,2],
                    color='c', alpha=pointalpha, s=pointsize)
    ax[2,1].scatter(biased_female[:,1], biased_female[:,2],
                    color='m', alpha=pointalpha, s=pointsize)

    ax[0,0].set_title("Unbiased population")
    ax[0,1].set_title("Biased population")

    for i in range(2):
        ax[0,i].set(xlabel="Hours per week", ylabel="Yearly income",
                    ylim=[8e3, 80e3])
        ax[1,i].set(xlabel="Years of experience", ylabel="Yearly income",
                    ylim=[8e3, 80e3])
        ax[2,i].set(xlabel="Years of experience", ylabel="Hourly wage")

        ax[0,i].plot([min_hours*workdays,max_hours*workdays],
                     [threshold, threshold], 'r--')

        ax[1,i].plot([min_exp,max_exp],
                     [threshold, threshold], 'r--')

    fig.tight_layout()
    plt.savefig('data-statistics.pdf', dpi=300)
    plt.show()



if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--samples-pop', type=int,
                        help="# samples for the population model", default=10000)
    parser.add_argument('--samples-pred', type=int,
                        help="# samples for the predictive model", default=10000)
    parser.add_argument('--seed', type=int, help="Seed number", default=666)

    args = parser.parse_args()

    np.random.seed(args.seed)

    plot_data = []
    for fstr in ['unbiased', 'biased']:

        exp_folder = f'data_{fstr}/'

        # create data folder if not existing
        if not os.path.isdir(exp_folder):
            os.mkdir(exp_folder)

        pop_path = os.path.join(exp_folder, 'pop_data.json')
        pred_path = os.path.join(exp_folder, 'pred_data.json')

        # sample the population data
        pop_data = sample(args.samples_pop, unbiased=(fstr=='unbiased'))


        # sample the prediction data
        pred_data = sample(args.samples_pred,
                           unbiased=(fstr=='unbiased'))

        # compute the positive/negative sample ratio
        n0, n1 = np.unique(pred_data[:,-1], return_counts=True)[1]
        nmin, nmax = min(n0, n1), max(n0, n1)
        ratio = 2* (nmax - nmin) / (nmax + nmin)

        print(f"Neg/Pos examples ratio ({fstr}) = {ratio}")
        
        plot_data.append(pop_data)

        # 0 = female, 1 = hpw, 2 = yexp, 3 = hw, 5 = age, 6 = class/label

        pop_feats = ['female', 'hpw', 'yexp', 'hw']
        pop_data = pop_data[:, [0, 1, 2, 3]]

        pred_feats = ['hpw', 'yexp', 'hw', 'y']
        pred_data = pred_data[:, [1, 2, 3, 6]]

        Dataset(pop_feats,
                ['bool', 'real', 'real', 'real'],
                pop_data).dump(pop_path)

        Dataset(pred_feats,
                ['real', 'real', 'real', 'bool'],
                pred_data).dump(pred_path)

    plot_statistics(*plot_data)
    

    

    
    
    

    
