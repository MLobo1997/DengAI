import numpy as np

def generate_time_sequence(X, y, window):
    new_X = []
    new_y = []

    for idx1, _ in enumerate(X):
        sequence_x = []
        sequence_y = []
        for idx2, (line_2, target) in enumerate(zip(X[idx1:], y[idx1:])):
            if idx2 < window:
                sequence_x.append(line_2.tolist())
                sequence_y.append(target)
            else:
                break

        if idx2 == window:
            new_X.append(sequence_x)
            new_y.append(sequence_y)
        else:
            break
    
    return np.array(new_X), np.array(new_y)