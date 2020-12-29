import argparse
import numpy as np
import pandas as pd
import csv
import os

def main():
    '''
    Run this python script in the terminal
    $ python -w [weigth path] -r [test set path] -o [predict result path]
    '''
    parser = argparse.ArgumentParser(description='Predict the PM2.5 value on the tenth day')
    parser.add_argument('-w', '--weight', default='./weight.npy')
    parser.add_argument('-r', '--raw', default='./test.csv')
    parser.add_argument('-o', '--output', default='./submit.csv')
    args = parser.parse_args()
    raw_file = args.raw
    output_file = args.output
    weight = args.weight
    if os.path.exists(weight) and os.path.exists(raw_file):
        predict(weight, raw_file, output_file)
        print('Predict the PM2.5 value and save the result in [{}]'.format(output_file))
    else:
        print('can not find path')


def predict(weigth, raw_file, output_file):
    # Pretreatment
    data_tst= pd.read_csv(raw_file, header=None, encoding='big5')
    data_tst[data_tst=='NR'] = 0
    arr_tst = np.array(data_tst.iloc[:, 2:11]).astype(float)
    test_X = np.empty([240, 18*9])
    for i in range(240):
        test_X[i, :] = arr_tst[i*18:(i+1)*18, :].reshape(1,-1)
    # Normalization
    mean = np.mean(test_X, axis=0)
    std = np.std(test_X, axis=0)
    for i in range(test_X.shape[0]):
        for j in range(test_X.shape[1]):
            if std[j] != 0:
                test_X[i][j] = (test_X[i][j] - mean[j]) / std[j]
    # Predict
    w = np.load(weigth)
    test_X = np.concatenate((np.ones((test_X.shape[0], 1)), test_X), axis=1)
    pred_y = test_X @ w

    with open(output_file, mode='w', newline='') as submit_file:
        csv_writer = csv.writer(submit_file)
        header = ['id', 'value']
        csv_writer.writerow(header)
        for i in range(240):
            row = ['id_' + str(i), pred_y[i][0]]
            csv_writer.writerow(row)


if __name__ == "__main__":
    main()