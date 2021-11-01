import cv2
import tensorflow as tf
import sys
import numpy as np
from load_data import Dataset
#from tf.keras.models import load_model
from human_gaze import my_softmax, my_kld, NSS
import matplotlib.pyplot as plt
#from utils import read_gaze

def reshape_image(image):
    """Warp frames to 84x84 as done in the Nature paper and later work."""
    width = 84
    height = 84
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return frame / 255.0


customObjects = {
    'my_softmax': my_softmax,
    'my_kld': my_kld,
    'NSS':NSS
}

model_path = '/scratch/user/ravikt/small.h5'

agil = tf.keras.models.load_model(model_path, custom_objects=customObjects)

agil.summary()

#d = Dataset(sys.argv[1], sys.argv[2]) 
#sample = d.generate_data_for_gaze_prediction()

##d.load_predicted_gaze_heatmap(sys.argv[3]) 
sample = cv2.imread('/scratch/user/ravikt/rgb_4.png')
sample = reshape_image(sample)
sample = np.reshape(sample, (1,84, 84, 1))

print(sample.shape)

print("Predicting results...")
output=agil.predict(sample)
print("Prediction complete!")

print("Output has following shape")
print(output.shape)

output = np.squeeze(output, axis=0)
#output = output*255
#print(output.shape)
plt.imshow(np.squeeze(output))
plt.imsave('out_4.png', np.squeeze(output))
#output = output.astype('uint8')

#print(np.nonzero(output))
#output = cv2.convertScaleAbs(output, alpha=(255.0))
#cv2.imwrite('out.png', output*255)
#cv2.imwrite('out.png', output)
#print("Writing output to compressed output")
#np.savez_compressed('results', heatmap=output[:,:,:,0])
#print("Writing completed")

