import matplotlib.pyplot as plt
import numpy as np
import os
import sys

if __name__ == '__main__':
    result_file = sys.argv[1]
    save_dir = sys.argv[2]

    step = []
    loss = []
    example_sec = []
    sec_batch = []

    with open(result_file, 'r') as f:
        for line in f:
            words = line.split(' ')
            step.append(int(words[3].replace(',','')))
            loss.append(float(words[6]))
            example_sec.append(float(words[7].replace('(','')))
            sec_batch.append(float(9))
    
    fig = plt.figure()
    plt.title('')
    plt.ylabel('loss')
    plt.xlabel('step')
    plt.plot(step, loss, 'r-')
    fig.savefig(os.path.join(save_dir, 'step_loss.png'), bbox_inches='tight')
    
    fig = plt.figure()
    plt.title('')
    plt.ylabel('examples/sec')
    plt.xlabel('step')
    plt.plot(step, example_sec, 'r-')
    fig.savefig(os.path.join(save_dir, 'step_examples_sec.png'), bbox_inches='tight')

    fig = plt.figure()
    plt.title('')
    plt.ylabel('sec/batch')
    plt.xlabel('step')
    plt.plot(step, sec_batch, 'r-')
    fig.savefig(os.path.join(save_dir, 'step_sec_batch.png'), bbox_inches='tight')
