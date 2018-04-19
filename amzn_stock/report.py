from pathlib import Path
import json
import matplotlib.pyplot as plt

def main():
    with Path('./result/log').open() as f:
        logs = json.load(f)
    
    loss_train = [log['main/loss'] for log in logs]
    loss_test = [log['validation/main/loss'] for log in logs]

    plt.plot(loss_train, label='train')
    plt.plot(loss_test, label='test')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()