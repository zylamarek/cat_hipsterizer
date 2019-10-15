import os
import numpy as np
from keras.models import load_model

dirname = 'test'
bbs_model_name = '2019_10_14_22_16_22'
base_path = '../cat-dataset/data/clean/%s' % dirname

img_size = 224
file_list = sorted(os.listdir(base_path))

dataset = {
    'bbs': []
}

data_bbs = np.load('dataset/%s.npy' % dirname, allow_pickle=True)
x_bbs = np.array(data_bbs.item().get('imgs')).astype('float32') / 255.
bbs_model = load_model('models/%s.h5' % bbs_model_name)
pred_bbs = bbs_model.predict(x_bbs, verbose=1)

dataset['bbs'] = [pred_bb.flatten() for pred_bb in pred_bbs]

np.save('dataset/predicted_bbox_%s_%s.npy' % (bbs_model_name, dirname), np.array(dataset))
