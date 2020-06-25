import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import json

def plot_training(logdir):
    # data_file = './logs/training_hist.txt'
    import glob
    data_files = glob.glob(logdir+'*.txt')
    # print(data_files)
    losses = []
    rewards = []
    names = []
    for data_file in data_files:

        with open(data_file,'rb') as f:
            data = json.load(f)
        names.append(data_file.split('/')[-1].split('_')[0])
        losses.append(data['loss'])
        rewards.append(data['episode_reward'])

    plt.figure()
    for (loss,name) in zip(losses,names):
        plt.plot(np.arange(len(loss)),loss,label=name)
    plt.legend()
    plt.title('Loss vs episode')
    plt.savefig('./figures/loss.png',dpi=300)

    plt.figure()
    for (reward,name) in zip(rewards,names):
        plt.plot(np.arange(len(reward)),reward,label=name)
    plt.legend()
    plt.title('rewards vs episode')
    plt.savefig('./figures/rewards.png',dpi=300)


if __name__ == '__main__':
    import glob
    plot_training('./logs/')