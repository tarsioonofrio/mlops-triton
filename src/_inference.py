import argparse
import numpy as np
import os
import tensorrtserver.api as tapi
import tensorrtserver.api.model_config_pb2 as model_config
import cv2
import queue as q


def model_dtype_to_np(model_dtype):
    if model_dtype == model_config.TYPE_BOOL:
        return np.bool
    elif model_dtype == model_config.TYPE_INT8:
        return np.int8
    elif model_dtype == model_config.TYPE_INT16:
        return np.int16
    elif model_dtype == model_config.TYPE_INT32:
        return np.int32
    elif model_dtype == model_config.TYPE_INT64:
        return np.int64
    elif model_dtype == model_config.TYPE_UINT8:
        return np.uint8
    elif model_dtype == model_config.TYPE_UINT16:
        return np.uint16
    elif model_dtype == model_config.TYPE_FP16:
        return np.float16
    elif model_dtype == model_config.TYPE_FP32:
        return np.float32
    elif model_dtype == model_config.TYPE_FP64:
        return np.float64
    elif model_dtype == model_config.TYPE_STRING:
        return np.dtype(object)
    return None


def get_serving_model_config(url, model_name, data_format=None, verbose=False):
    """
    Check the configuration of a model to make sure it meets the
    requirements for an image classification network (as expected by
    this client)
    Args:
        url: trt model inference server url
        model_name: serving model name
        batch_size: input array batch size for inference in serving
        data_format: input image array format, NHWC or NCHW
    """
    protocol = tapi.ProtocolType.from_str("HTTP")
    ctx = tapi.ServerStatusContext(url, protocol, model_name, verbose)
    server_status = ctx.get_server_status()

    if model_name not in server_status.model_status:
        raise Exception("unable to get status for '" + model_name + "'")

    status = server_status.model_status[model_name]
    config = status.config

    if len(config.input) != 1:
        raise Exception("expecting 1 input, got {}".format(len(config.input)))
    if len(config.output) != 1:
        raise Exception("expecting 1 output, got {}".format(len(config.output)))

    server_input = config.input[0]
    server_output = config.output[0]

    for dim in server_output.dims:
        if dim == -1:
            raise Exception("variable-size dimension in model output not supported")

    # Variable-size dimensions are not currently supported.
    for dim in server_input.dims:
        if dim == -1:
            raise Exception("variable-size dimension in model input not supported")
    if data_format == "NCHW":
        n, c, h, w = server_input.dims
    else:
        n, h, w, c = server_input.dims
    return (server_input.name, server_output.name, n, c, h, w, model_dtype_to_np(server_input.data_type))


class TRTInferServingEngine():
    def __init__(self, url, model_name, model_version=1, data_format="NCHW", async_run=True, request_count=1):
        input_name, output_name, n, c, h, w, dtype = get_serving_model_config(url, model_name, data_format)
        protocol = tapi.ProtocolType.from_str("HTTP")
        self.ctx = tapi.InferContext(url, protocol, model_name, model_version, verbose=False, correlation_id=0,
                                     streaming=False)
        self.data_format = data_format
        self.input_name = input_name
        self.output_name = output_name
        self.input_c = c
        self.input_h = h
        self.input_w = w
        self.input_batch = n
        self.dtype = dtype
        self.async_run = async_run
        self.request_count = request_count

    def preprocess(self, img):
        """
        Pre-process an image to meet the size, type and format
        requirements specified by the parameters.
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_w, self.input_h)).astype(self.dtype)
        # Swap to CHW if necessary
        if self.data_format == "NCHW":
            img = np.transpose(img, (2, 0, 1))
        return img

    def predict(self, img):
        # image after processing
        img = self.preprocess(img)
        # Send requests of FLAGS.batch_size images. If the number of
        # images isn't an exact multiple of FLAGS.batch_size then just
        # start over with the first images until the batch is filled.
        results = []
        request_ids = q.Queue()
        image_idx = 0
        last_request = False
        img_count = len(img)

        def completion_callback(ctx, request_id):
            q.put(request_id)

        while not last_request:
            input_batch = []
            for idx in range(self.input_batch):
                input_batch.append(img[image_idx])
                image_idx = (image_idx + 1) % img_count
                if image_idx == 0:
                    last_request = True
            # Send request
            if not self.async_run:
                results.append(self.ctx.run(
                    {self.input_name: [np.array(input_batch)]},
                    {self.output_name: tapi.InferContext.ResultFormat.RAW},
                    batch_size=1
                ))
            else:
                self.ctx.async_run_with_cb(
                    completion_callback,
                    {self.input_name: input_batch},
                    {self.output_name: tapi.InferContext.ResultFormat.RAW},
                    batch_size=1)

        # For async, retrieve results according to the send order
        r_count = 0
        if self.async_run:
            while True:
                request_id = request_ids.get()
                results.append(self.ctx.get_async_run_results(request_id, True))
                r_count += 1
                if r_count == self.request_count:
                    break
        return np.array(list(results[0].values())[0])[0]


if __name__ == "__main__":
    url = "172.18.0.1:8000"  # 172.18.0.1 is the docker ip
    # or
    # url = "trt_server:8000"
    model_name = "onnx_model"
    model_version = 1
    server_eng = TRTInferServingEngine(url, model_name, model_version)
    inputs = np.random.random((1, 3, 112, 112)).astype(np.float32)
    res = server_eng.predict(inputs)
    print(res)