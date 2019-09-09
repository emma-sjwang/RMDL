import os
import numpy as np
import sys
import random

def generate_testdata(total_feature, feature_path, local_feature_size=[150,512], shuffle=1, noise=True):
    filelist = os.listdir(feature_path)
    data_batch = np.ones([len(filelist), local_feature_size[0], local_feature_size[1]], dtype='float32')

    for index in xrange(len(filelist)):
        test_feature_path = os.path.join(feature_path, filelist[index])
        data = np.load(test_feature_path)[0 : local_feature_size[0]]
        max = np.amax(data)
        min = np.amin(data)
        data = 2.0 * (data - min) / (max - min + 0.00000001) - 1.0

        tempdata = data

        if shuffle:
            ll = np.random.permutation(local_feature_size[0])
        else:
            ll = range(total_feature)
        data_batch[index, :, :] = tempdata[ll[0:local_feature_size[0]]]
        # index += 1
    data_batch.astype('float32')
    return np.asarray(data_batch), filelist
