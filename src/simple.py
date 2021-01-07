# https://github.com/triton-inference-server/server/issues/1989
import numpy as np
from PIL import Image

from tritonhttpclient import InferenceServerClient, InferInput

# if run in docker container
# filename = "/workspace/images/mug.jpg"
filename = "../server/qa/images/mug.jpg"
model_name = "densenet_onnx"

triton_client = InferenceServerClient("localhost:8000")
config = triton_client.get_model_config(model_name)

img = Image.open(filename)
img = img.resize((224, 224), Image.ANTIALIAS)
img = np.array(img).astype(np.float32)
img = np.rollaxis(img, 2, 0)
# img = np.random.rand(3, 224, 224).astype(np.float32)
inp = InferInput(config['input'][0]['name'], img.shape, 'FP32')
inp.set_data_from_numpy(img)

for i in range(11):
    response = triton_client.infer(model_name=model_name, inputs=[inp])
    print(response.get_output(config['input'][0]["format"]))
