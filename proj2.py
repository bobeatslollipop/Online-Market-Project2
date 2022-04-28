import math
import csv
import os
import random
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt


random.seed(1)
tickers = []

with open(os.path.dirname(__file__) + '/proj2.csv') as f:
    r = csv.reader(f)
    for row in r:
        tickers.append(row)


def EW(data, epsilon=0.1, h=1):
    n = len(data)
    # first index for round, second for choice. STARTS AT ROUND 1
    k = len(data[0])
    actions = [j for j in range(k)]

    V = [[0 for j in range(k)]]  # same indexing
    does = []

    for i in range(1, n):
        V.append([(data[i][j] + V[i-1][j]) for j in range(k)])

        pitemp = [(1+epsilon) ** (V[i-1][j] / h) for j in range(k)]
        denom = sum(pitemp)
        pitemp = [pitemp[j] / denom for j in range(k)]
        do = random.choices(actions, weights=pitemp, k=1)
        does.append(do[0])

    return does


def FTL(data):
    n = len(data)
    k = len(data[0])
    V = [[0 for val in data[0]]]
    does = []

    for i in range(1, n):
        V.append([data[i][j] + V[i-1][j] for j in range(k)])

        argmax = 0
        for j in range(k):
            if V[i-1][j] == max(V[i-1]):
                argmax = j
                break
        does.append(argmax)

    return does


def generate_fair(n, k):
    data = [[0 for j in range(k)]]
    V = [[0 for val in data[0]]]
    for i in range(1, n):
        val = random.random()
        argmin = 0
        for j in range(k):
            if V[i-1][j] == min(V[i-1]):
                argmin = j
                break
        data.append([0 for j in range(k)])
        data[i][argmin] = val
        V.append([V[i-1][j] for j in range(k)])
        V[i][argmin] += val

    # plt.figure()
    # plt.plot(V[-1])
    # plt.xlabel("action")
    # plt.ylabel("emperical payoff")
    # plt.title("fair model, {} rounds".format(n))
    # plt.ylim(bottom=0)
    # plt.savefig('fair.png')

    return data


def generate_Bern(n, k):
    prob = [(j+1)/(2*k) for j in range(k)]
    data = [[0 for j in range(k)]]
    V = [[0 for val in data[0]]]
    for i in range(1, n):
        data.append([])
        for j in range(k):
            bern = random.choices([0, 1], weights=[1-prob[j], prob[j]])
            data[-1].append(bern[0])
        V.append([(V[-1][j] + data[-1][j]) for j in range(k)])

    # plt.figure()
    # plt.plot(V[-1])
    # plt.xlabel("action")
    # plt.ylabel("emperical payoff")
    # plt.title("Bernoulli model, {} rounds".format(n))
    # plt.savefig('Bern.png')

    return data


def get_stock_data(k, start='2020-03-31', end='2022-03-31'):
    ticker_n = len(tickers)
    inds = random.sample([i for i in range(ticker_n)], k)
    chosen = []
    raw = []
    # can use "len" and indexing to get price
    for ind in inds:
        raw.append(yf.download(tickers[ind][0], start, end))
        chosen.append(tickers[ind][0])
        # shape of raw is in (stock, type(Open/Close), day)

    n = max([len(raw[j]) for j in range(k)])  # number of days

    data = [[0 for j in range(k)]]
    V = [[0 for val in data[0]]]
    for i in range(1, n):
        data.append([])
        for j in range(k):
            if len(raw[j]) > i:
                data[i].append(raw[j]['Close'][i] / raw[j]['Open'][i])
                if data[i][-1] == float('nan'):
                    print(1/0)
            else:
                # if data absent, fill with 1 (stock value unchanged)
                data[i].append(1)
        V.append([(V[-1][j] + data[-1][j]) for j in range(k)])

    # plt.figure()
    # plt.plot(chosen, V[-1], label='{} rounds'.format(n))
    # plt.legend()
    # plt.xlabel("ticker")
    # plt.ylabel("emperical payoff")
    # plt.title("Stock, {} rounds".format(n))
    # plt.savefig('stock1.png')

    # plt.figure()
    # plt.plot(chosen, V[n // 8], label='{} rounds'.format(n//8))
    # plt.legend()
    # plt.xlabel("ticker")
    # plt.ylabel("emperical payoff")
    # plt.title("Stock, {} rounds".format(n//8))
    # plt.savefig('stock2.png')

    return chosen, data


def generate_adv(n, k):
    data = [[0 for j in range(k)]]
    V = [[0 for val in data[0]]]
    fair = generate_fair(n, k)
    Bern = generate_Bern(n, k)
    for i in range(1, n//2):
        data.append([0 for j in range(k)])
        data[i][i % 2] = 1
        V.append([(V[-1][j] + data[-1][j]) for j in range(k)])

    for i in range(n//2, n):
        data.append([0 for j in range(k)])
        data[i][0] = 1
        V.append([(V[-1][j] + data[-1][j]) for j in range(k)])

    # plt.figure()
    # plt.plot(V[-1], label='{} rounds'.format(n))
    # plt.legend()
    # plt.xlabel("action")
    # plt.ylabel("emperical payoff")
    # plt.title("Adversarial, {} rounds".format(n))
    # plt.savefig('adv1.png')

    # plt.figure()
    # plt.plot(V[n // 2], label='{} rounds'.format(n//2))
    # plt.legend()
    # plt.xlabel("action")
    # plt.ylabel("emperical payoff")
    # plt.title("Adversarial, {} rounds".format(n//2))
    # plt.savefig('adv2.png')

    return data


def analyze_moves(data, moves):
    n = len(data)
    k = len(data[0])

    OPT = 0
    maxreturn = 0
    for j in range(k):
        total = 0
        for i in range(n):
            total += data[i][j]
        if maxreturn < total:
            maxreturn = total
            OPT = j

    curreturn = 0
    for i in range(1, n):
        curreturn += data[i][moves[i-1]]
    regret = (maxreturn - curreturn) / (n-1)

    return OPT, regret


def test_1(N=20, d=30, n=100, k=10):
    epsilon_list = [i/10 for i in range(d-1)]
    epsilon_list.append(math.sqrt(math.log(k) / n))
    fair_OPT = np.zeros((N, d+1))
    fair_regret = np.zeros((N, d+1))
    Bern_OPT = np.zeros((N, d+1))
    Bern_regret = np.zeros((N, d+1))

    for i in range(N):
        fair_data = generate_fair(n, k)
        Bern_data = generate_Bern(n, k)

        for e in range(d):
            fair_OPT[i][e], fair_regret[i][e] = analyze_moves(
                fair_data, EW(fair_data, epsilon=epsilon_list[e]))
            Bern_OPT[i][e], Bern_regret[i][e] = analyze_moves(
                Bern_data, EW(Bern_data, epsilon=epsilon_list[e]))

        # Custom built FTL algorithm as comparison
        fair_OPT[i][d], fair_regret[i][d] = analyze_moves(
            fair_data, FTL(fair_data))
        Bern_OPT[i][d], Bern_regret[i][d] = analyze_moves(
            Bern_data, FTL(Bern_data))

    fair_avg_reg = np.mean(fair_regret, axis=0)
    Bern_avg_reg = np.mean(Bern_regret, axis=0)
    epsilon_list[-1] = 'opt'
    epsilon_list.append('inf')
    plt.figure()
    plt.plot(epsilon_list, fair_avg_reg, label='fair')
    plt.plot(epsilon_list, Bern_avg_reg, label='bern')
    plt.legend()
    plt.xlabel("epsilon")
    plt.ylabel("average regret")
    plt.title("n={}".format(n))

    plt.savefig('test1.png')


def test_2(N=20, d=30, n=100, k=10):
    epsilon_list = [i/5 for i in range(d-1)]
    epsilon_list.append(math.sqrt(math.log(k) / n))
    stock_OPT = np.zeros((N, d+1))
    stock_regret = np.zeros((N, d+1))
    adv_OPT = np.zeros((N, d+1))
    adv_regret = np.zeros((N, d+1))

    for i in range(N):
        # year = random.randint(1980, 2021)
        # start = "{}-01-01".format(year)
        # end = "{}-01-01".format(year+1)
        # stock_data = get_stock_data(start, end)
        chosen, stock_data = get_stock_data(k)
        adv_data = generate_adv(n, k)

        for e in range(d):
            stock_OPT[i][e], stock_regret[i][e] = analyze_moves(
                stock_data, EW(stock_data, epsilon=epsilon_list[e], h=2))
            adv_OPT[i][e], adv_regret[i][e] = analyze_moves(
                adv_data, EW(adv_data, epsilon=epsilon_list[e]))

        # Custom built FTL algorithm as comparison
        stock_OPT[i][d], stock_regret[i][d] = analyze_moves(
            stock_data, FTL(stock_data))
        adv_OPT[i][d], adv_regret[i][d] = analyze_moves(
            adv_data, FTL(adv_data))

    stock_avg_reg = np.mean(stock_regret, axis=0)
    adv_avg_reg = np.mean(adv_regret, axis=0)
    epsilon_list[-1] = 'opt'
    epsilon_list.append('inf')
    plt.figure()
    plt.plot(epsilon_list, stock_avg_reg, label='stock')
    # plt.plot(epsilon_list, adv_avg_reg, label='adv')
    plt.xlabel("epsilon")
    plt.ylabel("average regret")
    plt.title("Stock n={}".format(len(stock_data)))
    plt.savefig('stock.png')

    plt.figure()
    plt.plot(epsilon_list, adv_avg_reg, label='adv')
    plt.xlabel("epsilon")
    plt.ylabel("average regret")
    plt.title("Adversarial n={}".format(n))
    plt.savefig('adv.png')


test_2(40, k=10, d=15, n=500)
