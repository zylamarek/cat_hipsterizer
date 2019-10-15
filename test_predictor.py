import numpy as np
from keras.models import load_model
from collections import defaultdict

dirname = 'test'
bbs_model_name = '2019_10_14_22_16_22'
lmks_model_name = '2019_10_14_23_50_08'

img_size = 224

data_lmks = np.load('dataset/lmks_pred_%s_%s.npy' % (bbs_model_name, dirname), allow_pickle=True)
x_lmks = np.array(data_lmks.item().get('imgs')).astype('float32') / 255.
y_lmks = np.array(data_lmks.item().get('lmks'))
landmarks_originals = np.array(data_lmks.item().get('landmarks_original'))
lmks_transforms = np.array(data_lmks.item().get('transform'))
img_max_sizes = np.array(data_lmks.item().get('img_max_size'))

lmks_model = load_model('models/%s.h5' % lmks_model_name)
predicted_lmks = lmks_model.predict(x_lmks, verbose=1)
mse_eval = lmks_model.evaluate(x_lmks, y_lmks, verbose=1)

mses = defaultdict(list)
for landmarks_predicted, lmks_transform, landmarks_original, img_max_size in \
        zip(predicted_lmks, lmks_transforms, landmarks_originals, img_max_sizes):
    landmarks_predicted = landmarks_predicted.reshape((-1, 2)) - np.array(lmks_transform[1:3])
    landmarks_predicted /= lmks_transform[0]
    landmarks_predicted += lmks_transform[3]
    landmarks_predicted = landmarks_predicted
    landmarks_original = landmarks_original.reshape((-1, 2))


    def get_mse_normalized(a, b):
        return np.mean(np.square((landmarks_predicted[a: b + 1] - landmarks_original[a: b + 1]) / img_max_size))


    mse = np.mean(np.square(landmarks_predicted - landmarks_original))
    mses['all'].append(mse)
    mses['all normalized'].append(get_mse_normalized(0, 8))
    mses['eyes normalized'].append(get_mse_normalized(0, 1))
    mses['mouth normalized'].append(get_mse_normalized(2, 2))
    mses['ears normalized'].append(get_mse_normalized(3, 8))

print('mse all: %.7f' % np.mean(mses['all']))
for name, vals in mses.items():
    print('rmse %s: %.7f' % (name, np.sqrt(np.mean(vals))))
print('mse eval', mse_eval)
print('rmse eval', np.sqrt(mse_eval))
