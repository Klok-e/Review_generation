import multiprocessing.pool
import gzip
import keras.callbacks
import numpy as np
import OneHotEncoder
from keras.layers import LSTM, Dense, Dropout, TimeDistributed
from keras.models import Sequential

path = 'reviews_Amazon_Instant_Video_5.json.gz'


def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)
    g.close()


def lstm_model(one_hot_length: int, timestep: int, training: bool, weights=None):
    model = Sequential()

    model.add(LSTM(256, batch_input_shape=(None if training else 1, timestep, one_hot_length), return_sequences=True,
                   stateful=not training))
    model.add(Dropout(0.2))
    model.add(LSTM(256, return_sequences=True, stateful=not training))
    model.add(Dropout(0.2))
    model.add(LSTM(200, return_sequences=True, stateful=not training))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(one_hot_length, activation='softmax')))
    if (weights):
        model.load_weights(weights)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


def get_reviews(path):
    data = list(parse(path))
    return [review['reviewText'] for review in data]


def get_one_hot(path):
    reviews = get_reviews(path)
    return OneHotEncoder.OneHotCharacter(list(set([char for review in reviews for char in review])))


def resh(xx, args):
    return np.reshape(np.array(xx, np.int8), args)


def main():
    seq_length = 200

    reviews = get_reviews(path)
    one_hot = get_one_hot(path)

    seq_x = []
    seq_y = []
    for review in reviews:
        seq_x.extend([one_hot[char] for char in review])
        seq_y.extend([one_hot[char] for char in review[1:]] + [one_hot.end_seq_vec, ])
    del reviews

    # cut sequences
    assert len(seq_x) == len(seq_y)
    samples_x = [seq_x[i:i + seq_length] for i in range(len(seq_x) // seq_length)]
    samples_y = [seq_y[i:i + seq_length] for i in range(len(seq_y) // seq_length)]
    del seq_x, seq_y

    # some parallel computing
    my_pool = multiprocessing.pool.Pool(processes=2)
    xy = [my_pool.apply_async(resh, (samples_x, (len(samples_x), seq_length, len(one_hot)))),
          my_pool.apply_async(resh, (samples_y, (len(samples_y), seq_length, len(one_hot))))]
    del samples_x, samples_y

    model = lstm_model(len(one_hot), seq_length, training=True)

    x, y = [res.get() for res in xy]
    my_pool.close()

    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    model.fit(x, y, batch_size=256, epochs=20, callbacks=callbacks_list)
    model.save_weights(path)


if __name__ == '__main__':
    main()
