import tensorflow as tf
import sys
from load_data import Dataset
#from tf.keras.models import load_model
from human_gaze import my_softmax, my_kld, NSS

customObjects = {
    'my_softmax': my_softmax,
    'my_kld': my_kld,
    'NSS':NSS
}

agil = tf.keras.models.load_model('trained_gaze_models/alien.hdf5', custom_objects=customObjects)

agil.summary()

d = Dataset(sys.argv[1], sys.argv[2]) 
sample = d.generate_data_for_gaze_prediction()
#d.load_predicted_gaze_heatmap(sys.argv[3]) 

agil.predict(sample, 4)