# import streamlit as st
# import tensorflow as tf
# from PIL import Image
# import pickle
# with open("categories.pickle", 'rb') as f:
#     categories = pickle.load(f)

# # Load your TensorFlow model
# model = tf.keras.models.load_model("model.h5")

# # Create a file uploader widget
# uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# # Show the image
# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image")
#     prediction = "none"
#     # Use the model to classify the image
#     prediction = model.predict(image)
#     st.write("Prediction: ", prediction)


import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle
with open("categories.pickle", 'rb') as f:
    categories = pickle.load(f)

st.set_option('deprecation.showfileUploaderEncoding', False)


@st.cache(allow_output_mutation=True)
def load_model():
	model = tf.keras.models.load_model('model.h5')
	return model


def predict_class(image, model):

	image = tf.cast(image, tf.float32)
	image = tf.image.resize(image, [224, 224])
	image = np.reshape(image,(1,224,224,-1))

	prediction = model.predict(image)

	return prediction


model = load_model()

st.title('satelite image clssifier')

file = st.file_uploader("Upload an image of a satellite", type=["jpg", "png"])


if file is None:
	st.text('Waiting for upload....')

else:
	slot = st.empty()
	slot.text('Running inference....')
	test_image = Image.open(file).convert("RGB")
	st.image(test_image, caption="Input Image", width=400)

	pred = predict_class(np.asarray(test_image), model)

	result = categories[np.argmax(pred)]

	output = 'The image is a ' + result

	slot.text('Done')

	st.success(output)
