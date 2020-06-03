import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import json

def plot_training(data_file):
    # data_file = './logs/training_hist.txt'
    with open(data_file,'rb') as f:
        data = json.load(f)

    losses = data['loss']
    rewards = data['episode_reward']

    plt.figure()
    plt.plot(np.arange(len(losses)),losses)
    plt.title('Loss vs episode')
    plt.savefig('./figures/loss.png',dpi=300)


    plt.figure()
    plt.plot(np.arange(len(rewards)),rewards)
    plt.title('rewards vs episode')
    plt.savefig('./figures/rewards.png',dpi=300)