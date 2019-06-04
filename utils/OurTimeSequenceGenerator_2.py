import numpy as np

def generate_time_sequence2(X, y, window, previous_n, nan_val=-999):
    new_X = []
    new_y = []

    total = window + previous_n

    for idx1, _ in enumerate(X):
        sequence_x = []
        sequence_y = []
        for idx2, (line_2, target) in enumerate(zip(X[idx1:], y[idx1:])):
            if idx2 < total:
                new_line = line_2.tolist()
                if idx2 < previous_n:
                    new_line.append(target)
                else:
                    new_line.append(nan_val)
                    sequence_y.append(target)
                sequence_x.append(new_line)
            else:
                break

        if idx2 == total:
            new_X.append(sequence_x)
            new_y.append(sequence_y)
        else:
            break
    
    return np.array(new_X), np.array(new_y)