from keras.models import *
from keras.regularizers import l2
from keras.layers import Input, Dense, Concatenate, Dropout, MaxPooling1D, LeakyReLU, Softmax, Multiply, Flatten
import time
import sys
sys.path.append("./2detector/Code/mil_nets/")
from mil_nets.layer import Expand_dims, ADD
sys.path.append("./")
from load_data import *
from keras import optimizers
from keras.utils.np_utils import to_categorical
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


def write2results(filename_list, predictions, probability):
    predict_wrong_log = './Outputs/results.csv'

    fp = open(predict_wrong_log, 'wt')
    fp.write("filename prediction confidence\n")
    label_list = ['normal', 'dysplasia', 'cancer']

    for i in range(predictions.shape[0]):
        label_index = list(predictions[i]).index(1)
        fp.write("%s %s %f\n" % (filename_list[i].split('.')[0], label_list[label_index], probability[i][label_index]))
        print('File %s is predicted as %s, with a confidence of %f.'
              %(filename_list[i].split('.')[0], label_list[label_index], probability[i][label_index]))
    fp.close()

class CHOWDER(object):
    def __init__(self, total_feature=150, n_class=2, lr=1e-3, epochs=500, train_iters=5,
                 batch_size=128, every_epochs=1, GPU_ID=0, Load_weights_from= "", feature_path="",
                 local_feature_size=[], global_feature_size=[], shuffle = False):
        self.total_feature = total_feature
        self.epochs = epochs
        self.every_epochs = every_epochs
        self.train_iters = train_iters
        self.batch_size = batch_size
        self.n_class = n_class
        self.first_load = True
        self.GPU_ID = GPU_ID
        self.lr = lr
        self.Load_weights_from = Load_weights_from
        self.feature_path = feature_path
        self.local_feature_size = local_feature_size
        self.global_feature_size = global_feature_size
        self.shuffle = shuffle

        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.GPU_ID)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

    def get_chowder(self):

        data_input = Input(shape=(local_feature_size[0], local_feature_size[1],), dtype='float32', name='input')
        fc1 = Dense(128, kernel_regularizer=l2(0.005), name='fc1')(data_input)
        fc1 = InstanceNormalization(axis=-1)(fc1)
        fc1 = LeakyReLU(alpha=0.2)(fc1)
        fc1 = Dropout(rate=0.5)(fc1)

        fc2 = Dense(64, kernel_regularizer=l2(0.005), name='fc2')(fc1)
        fc2 = InstanceNormalization(axis=-1)(fc2)
        fc2 = LeakyReLU(alpha=0.2)(fc2)
        fc2 = Dropout(rate=0.5)(fc2)
        fc3 = Dense(64, kernel_regularizer=l2(0.005), name='fc3')(fc2)
        fc3 = InstanceNormalization(axis=-1)(fc3)
        fc3 = LeakyReLU(alpha=0.2)(fc3)
        fc3 = Dropout(rate=0.5)(fc3)

        gf1 = MaxPooling1D(pool_size=self.local_feature_size[0], name='gf1')(fc1)
        gf2 = MaxPooling1D(pool_size=self.local_feature_size[0], name='gf2')(fc2)

        gf = Concatenate(name='concatenate')([gf1, gf2])
        gf = Expand_dims(dims=local_feature_size[0])(gf)
        feature_combine = Concatenate(axis=-1, name='concatenate_gl')([fc3, gf])
        weights = Dense(1, kernel_regularizer=l2(0.005), name='weights')(feature_combine)
        weights = Softmax(axis=-2, name='weight')(weights)
        feature_combine = Multiply()([weights, data_input])
        mp = ADD(name='mip')(feature_combine)

        mp = LeakyReLU(alpha=0.2)(mp)
        mp = Dropout(rate=0.1)(mp)
        mp = Flatten()(mp)

        output = Dense(self.n_class, kernel_regularizer=l2(0.005), activation='softmax', name='output')(mp)
        model = Model(inputs=[data_input], outputs=[output])
        adam = optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        self.model = model

    def predict(self):
        self.with_noise = 0
        self.get_chowder()
        if os.path.exists(self.Load_weights_from):
            print "==> Load model weights from %s \n" % self.Load_weights_from
            self.model.load_weights(self.Load_weights_from, by_name=True)
        else:
            print "!ERROR: CANNOT load weights file %s." % self.Load_weights_from

        test_data, filename_list = generate_testdata(self.total_feature,
                                                        self.feature_path, self.local_feature_size,
                                                        self.n_class, self.shuffle)
        predictions = np.zeros([len(filename_list), 3])
        result = self.model.predict(test_data, batch_size=self.batch_size)

        for i in range(len(result)):
            predictions[i, :] = to_categorical(np.argmax((result[i, :]), axis=-1), self.n_class)

        write2results(filename_list, predictions, result)
        return

if __name__ == '__main__':

    shuffle = True
    _EPSILON = 1e-7

    GPU_ID = int(sys.argv[1])
    total_feature = 100
    local_feature_size = [300, 1024]
    global_feature_size = [102]
    print "==> Start Prediction"
    feature_path = "./Outputs/ROIs/features/npyDir/"
    Load_weights_from = "./weights/RMDL.h5"
    chowder = CHOWDER(total_feature=total_feature, n_class=3, lr=1e-3, epochs=100, train_iters=5, batch_size=256, every_epochs=1,
                      GPU_ID=GPU_ID, Load_weights_from=Load_weights_from,
                      feature_path=feature_path,
                      local_feature_size=local_feature_size, global_feature_size=global_feature_size, shuffle=shuffle,
                      )
    chowder.predict()
