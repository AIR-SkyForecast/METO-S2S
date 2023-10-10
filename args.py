import argparse


def args_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--epochs', type=int, default=30, help='input dimension')
    parser.add_argument('--batch_size', type=int, default=16, help='input dimension')
    parser.add_argument('--input_size', type=int, default=1, help='input dimension')
    parser.add_argument('--seq_len', type=int, default=12, help='seq_len')
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden_size')
    parser.add_argument('--bidirectional', type=bool, default=False, help='LSTM direction')
    parser.add_argument('--file_name', type=str, default='.data/data.csv', help='LSTM direction')
    return parser.parse_args()

if __name__ == '__main__':
    args = args_parser()
    print("args:", args)
