import main as m
import numpy as np


def sample(x, temp=1.):
    a = np.array(x) ** (1 / temp)
    p_sum = a.sum()
    sample_temp = a / p_sum
    return sample_temp


def generate(weights):
    one_hot = m.get_one_hot(m.path)
    pattern = [one_hot[char] for char in 'A ']

    model = m.lstm_model(len(one_hot), 1, weights=weights, training=False)

    generated = ''
    for i in range(1000):
        x = m.np.reshape(pattern[i], (1, 1, len(one_hot)))
        prediction = sample(model.predict(x, verbose=0),0.1)

        ind = np.random.choice(np.arange(0, len(one_hot), 1), p=np.reshape(prediction, (len(one_hot))))

        next_char_vec = [0 for i in range(len(one_hot))]
        next_char_vec[ind] = 1
        next_char_vec = tuple(next_char_vec)

        try:
            pattern[i + 1]
        except IndexError:
            pattern.append(next_char_vec)

        generated += one_hot[next_char_vec]
    print(generated)


if __name__ == '__main__':
    generate('weights-improvement-20-0.5586.hdf5')
