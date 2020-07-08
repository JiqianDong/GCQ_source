import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import json

import json
import matplotlib.pyplot as plt
import numpy as np

def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed

def plot_training(logdir,loss_smooth_weight=0.3,reward_smooth_weight=0.85):
    import glob
    data_files = glob.glob(logdir+'*training_hist.txt')
    losses = []
    losses_smoothed = []
    rewards = []
    rewards_smoothed = []
    names = []
    for data_file in data_files:

        with open(data_file,'rb') as f:
            data = json.load(f)
        names.append(data_file.split('/')[-1].split('_')[0])


        loss = np.array(data['loss'])
        reward = np.array(data['episode_reward'])
        loss = loss[~np.isnan(loss)] #drop nan for smoothing

        loss_smoothed = smooth(loss, loss_smooth_weight)
        reward_smoothed = smooth(reward,reward_smooth_weight)

        losses.append(loss)
        rewards.append(reward)

        losses_smoothed.append(loss_smoothed)
        rewards_smoothed.append(reward_smoothed)


    plt.figure()
    for (loss,loss_smoothed,name) in zip(losses,losses_smoothed,names):
        p = plt.plot(np.arange(len(loss)),loss,label=name,alpha=0.2)
        plt.plot(np.arange(len(loss_smoothed)),loss_smoothed,label=name,c=p[0].get_color())
    plt.legend()
    plt.title('Loss vs episode')
    plt.savefig('./figures/loss.png',dpi=300)

    plt.figure()

    for (reward,reward_smoothed,name) in zip(rewards,rewards_smoothed,names):

        p = plt.plot(np.arange(len(reward)),reward,alpha=0.2)

        plt.plot(np.arange(len(reward_smoothed)),reward_smoothed,label=name,c=p[0].get_color())
    plt.legend()
    plt.title('rewards vs episode')
    plt.savefig('./figures/rewards.png',dpi=300)





if __name__ == '__main__':
    import glob
    plot_training('./logs/',0.3,0.85)