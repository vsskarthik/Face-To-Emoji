import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
print('Tensorlow Loaded\n\n')

models = { 	0: '../trained_models/fer_0-1426_0-9519.h5',
		   	1: '../trained_models/fer_0-1042_0-9651.h5',
			2: '../trained_models/fer_0-0268_0-9910.h5'}

model = tf.keras.models.load_model(models[0])
print("Model Summary:\n\n")
print(model.summary())