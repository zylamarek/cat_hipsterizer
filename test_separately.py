import numpy as np
from keras.models import load_model
import keras.backend as K

dirname = 'test'
dirname_landmarks = 'landmarks_test'
bbs_model_name = '2019_10_14_22_16_22'
lmks_model_name = '2019_10_14_23_50_08'
img_size = 224

data_bbs = np.load('dataset/%s.npy' % dirname, allow_pickle=True)
x_bbs = np.array(data_bbs.item().get('imgs')).astype('float32') / 255.
y_bbs = np.array(data_bbs.item().get('bbs'))

data_lmks = np.load('dataset/lmks_%s.npy' % dirname_landmarks, allow_pickle=True)
x_lmks = np.array(data_lmks.item().get('imgs')).astype('float32') / 255.
y_lmks = np.array(data_lmks.item().get('lmks'))


def iou_metric(y_true, y_pred):
    y_true = K.permute_dimensions(y_true, (1, 0))
    y_pred = K.permute_dimensions(y_pred, (1, 0))

    x_0 = K.max([K.gather(y_true, 0), K.gather(y_pred, 0)], axis=0)
    y_0 = K.max([K.gather(y_true, 1), K.gather(y_pred, 1)], axis=0)
    x_1 = K.min([K.gather(y_true, 2), K.gather(y_pred, 2)], axis=0)
    y_1 = K.min([K.gather(y_true, 3), K.gather(y_pred, 3)], axis=0)

    area_inter = K.clip(x_1 - x_0, 0, None) * K.clip(y_1 - y_0, 0, None)

    area_true = (K.gather(y_true, 2) - K.gather(y_true, 0)) * (K.gather(y_true, 3) - K.gather(y_true, 1))
    area_pred = (K.gather(y_pred, 2) - K.gather(y_pred, 0)) * (K.gather(y_pred, 3) - K.gather(y_pred, 1))

    iou_ = area_inter / (area_true + area_pred - area_inter)

    return K.mean(iou_, axis=-1)


bbs_model = load_model('models/%s.h5' % bbs_model_name)
bbs_model.compile(optimizer='sgd', loss='mse', metrics=['mae', iou_metric])
bbs_mse, bbs_mae, bbs_iou = bbs_model.evaluate(x_bbs, y_bbs)

lmks_model = load_model('models/%s.h5' % lmks_model_name)
lmks_model.compile(optimizer='sgd', loss='mse', metrics=['mae'])
lmks_mse, lmks_mae = lmks_model.evaluate(x_lmks, y_lmks)

print('bbs IoU:\t%.2f' % (bbs_iou * 100.))
print('bbs MAE:\t%.2f' % bbs_mae)
print('bbs MSE:\t%.2f' % bbs_mse)
print('bbs RMSE:\t%.2f' % np.sqrt(bbs_mse))

print('lmks MAE:\t%.2f' % lmks_mae)
print('lmks MSE:\t%.2f' % lmks_mse)
print('lmks RMSE:\t%.2f' % np.sqrt(lmks_mse))
