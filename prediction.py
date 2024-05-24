from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from fastapi import UploadFile, File

IMG_SIZE = 224

class CustomLayer(keras.layers.Layer):
    def __init__(self, sublayer, **kwargs):
        super().__init__(**kwargs)
        self.sublayer = sublayer

    def call(self, x):
        return self.sublayer(x)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "sublayer": keras.saving.serialize_keras_object(self.sublayer),
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        sublayer_config = config.pop("sublayer")
        sublayer = keras.saving.deserialize_keras_object(sublayer_config)
        return cls(sublayer, **config)
    
  
def load_model(model_path):
    """
    Loads a saved model from a specified path.
    """
    print(f"Loading saved model from: {model_path}")

    # Load the MobileNetV2 model but exclude the top layer (classification layer)
    feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
    feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                             input_shape=(224,224,3))

    # Wrap the KerasLayer in a Lambda layer
    wrapped_feature_extractor_layer = tf.keras.layers.Lambda(
        lambda x: feature_extractor_layer(x)
    )

    # Create a new model with the wrapped KerasLayer as the first layer
    model = "20240522-10471716374835-full-image-set-mobilenetv2-Adam.h5"

# Create a function to load a trained model
# def load_model(model_path):
#   """
#   Loads a saved model from a specified path.
#   """
#   print(f"Loading saved model from: {model_path}")
#   model = tf.keras.models.load_model(model_path,
#                                     custom_objects={"KerasLayer":hub.KerasLayer})
#   return model

model = load_model("20240522-10471716374835-full-image-set-mobilenetv2-Adam.h5")
  
async def read_image(file: UploadFile):
    try:
        image = ...
        if image is None:
            raise ValueError(f"Failed to read image from file {file.filename}")
        print(image)
        return image
    except Exception as e:
        print(f"Error reading image file {file.filename}: {e}")
        raise

# def read_image(image_encoded):
#     pil_Image = Image.open(BytesIO(image_encoded))
#     return pil_Image

def process_image(image_path):
  """
  Takes an image file path and turn the image into a Tensor.
  """
  # Read in an image file
  image = tf.io.read_file(image_path)
  # Turn the jpeg image into numerical Tensor with 3 colour channels(Res, Green, Blue)
  image = tf.image.decode_jpeg(image, channels=3)
  # Convert the colour channels values from 0-255 to 0-1 values
  image = tf.image.convert_image_dtype(image, tf.float32)
  # Resize the image to our desired value(224, 224)
  image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])

  return image

# def process_image(pil_image):
#     """
#     Takes a PIL Image object and turns it into a Tensor suitable for input into MobileNetV2.
#     """
#     # Convert the PIL Image to a NumPy array
#     image = np.array(pil_image)

#     # Ensure the image has 3 color channels (RGB)
#     if len(image.shape) == 2:
#         image = np.repeat(image[:, :, np.newaxis], 3, axis=2)

#     # Convert the image to a TensorFlow Tensor and resize it to the correct size
#     image = tf.convert_to_tensor(image)
#     image = tf.image.resize(image, [224, 224])

#     # Normalize the pixel values to be in the range [0, 1]
#     image = image / 255.0

#     # Add an extra dimension for the batch size
#     image = tf.expand_dims(image, axis=0)

#     return image



def predict(image : np.ndarray):
    model.predict(image)