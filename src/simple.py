# https://github.com/triton-inference-server/server/issues/1989

from tritonhttpclient import InferenceServerClient, InferInput
import numpy as np

model_name = "densenet_onnx"

triton_client = InferenceServerClient("localhost:8000")
config = triton_client.get_model_config(model_name)

img = np.random.rand(3, 224, 224).astype(np.float32)
inp = InferInput(config['input'][0]['name'], img.shape, 'FP32')
inp.set_data_from_numpy(img)

for i in range(11):
    response = triton_client.infer(model_name=model_name, inputs=[inp])
