import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import json

import json
import matplotlib.pyplot as plt
import numpy as np

line_type_dic = {"gcn":'-','lstm':':','rule_based':'--'}
line_width_dic = {"gcn":1,'lstm':2,'rule_based':1}

def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed


def plot_training(logdir,loss_smooth_weight=0.3,reward_smooth_weight=0.85,loss_y_lim=None, reward_y_lim=None):
    import glob
    data_files = glob.glob(logdir+'*training_hist.txt')
    losses = []
    losses_smoothed = []
    rewards = []
    rewards_smoothed = []
    names = []
    data_files.sort()
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
        p = plt.plot(np.arange(len(loss)),loss,alpha=0.2,linestyle=line_type_dic[name])
        plt.plot(np.arange(len(loss_smoothed)),loss_smoothed,label=name,c=p[0].get_color(),linestyle=line_type_dic[name],linewidth=line_width_dic[name])
    plt.legend()
    if loss_y_lim:
        plt.ylim(loss_y_lim)
    plt.title('Loss vs episode')
    plt.xlabel("episode (after training start)")
    plt.ylabel("loss")
    plt.savefig('./figures/loss.png',dpi=300,bbox_inches='tight')

    plt.figure()

    for (reward,reward_smoothed,name) in zip(rewards,rewards_smoothed,names):

        p = plt.plot(np.arange(len(reward)),reward,alpha=0.1)

        plt.plot(np.arange(len(reward_smoothed)),reward_smoothed,label=name,c=p[0].get_color(),linestyle=line_type_dic[name],linewidth=line_width_dic[name])


    name = "rule_based"
    plt.plot([0,len(reward_smoothed)-1],[-6281.482498+2000]*2,label=name, linestyle=line_type_dic[name])

    plt.legend()
    if reward_y_lim:
        plt.ylim(reward_y_lim)
    plt.title('rewards vs episode')
    plt.ylabel("episode reward")
    plt.xlabel("episode")
    plt.savefig('./figures/rewards.png',dpi=300,bbox_inches='tight')





if __name__ == '__main__':
    import glob
    plot_training('./logs/',0,0.9,(10,70),(-10000,3000))
    #plot_training('./logs/',0.3,0.9)