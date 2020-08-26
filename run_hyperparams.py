import numpy as np
import data_process
import train_model

# iterate over hyperparameters and store results
def main():
    layers = 3
    penalty_vals = range(30, 55, 5)
    epoch_vals = [1_000, 3_000, 10_000, 30_000, 100_000]
    lr_vals = [0.0001, 0.0003, 0.001, 0.003, 0.01]

    params_list = ['Epochs', 'Alpha', 'AvgTestLoss', 'StdTestLoss', 'AvgLoss/DataStd']

    num_trials = 10
    
    with open('loss_3_layers.txt', 'a') as f:
        f.write(f'Layers: {layers}')

        for pen_val in penalty_vals:

            f.write(f'\n\n\nPenalty {pen_val}\n')
            results_avg, results_std = data_process.change_penalty(pen_val)
            f.write(f'Data average: {results_avg:.5f}\n')
            f.write(f'Data standard dev: {results_std:.5}\n\n')

            # column labels
            for param in params_list:
                f.write(f'{param:15}')
            f.write('\n')

            for eps in epoch_vals:
                for lr in lr_vals:

                    f.write(f'{eps:<15d}')
                    f.write(f'{lr:<15.4f}')

                    # average the test loss over several trials
                    test_loss = []
                    for i in range(num_trials):
                        loss = train_model.main(pen_val, lr, eps)
                        test_loss.append(loss)
                    
                    test_loss = np.asarray(test_loss)
                    test_loss_avg, test_loss_std = np.mean(test_loss), np.std(test_loss)
                    test_loss_norm = test_loss_avg / results_std
                    f.write(f'{test_loss_avg:<15.5f}  ')
                    f.write(f'{test_loss_std:<15.5f}')
                    f.write(f'{test_loss_norm:<15.5f}\n')

if __name__ == "__main__":
    main()