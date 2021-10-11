import tensorflow as tf
import sys
import numpy as np
from load_data import Dataset
#from tf.keras.models import load_model
from human_gaze import my_softmax, my_kld, NSS

customObjects = {
    'my_softmax': my_softmax,
    'my_kld': my_kld,
    'NSS':NSS
}

agil = tf.keras.models.load_model('alien.hdf5', custom_objects=customObjects)

agil.summary()

d = Dataset(sys.argv[1], sys.argv[2]) 
sample = d.generate_data_for_gaze_prediction()
#d.load_predicted_gaze_heatmap(sys.argv[3]) 

print("Predicting results...")
output=agil.predict(sample)
print("Prediction complete!")

print("Output has following shape")
print(output.shape)

print("Writing output to compressed output")
np.savez_compressed('results', heatmap=output[:,:,:,0])
print("Writing completed")

