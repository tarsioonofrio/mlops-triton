#!/usr/bin/env python
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import sys
from functools import partial
import os
import queue

import numpy as np
from PIL import Image

import tritonclient.grpc as grpcclient
import tritonclient.grpc.model_config_pb2 as mc
import tritonclient.http as httpclient
from tritonclient.utils import triton_to_np_dtype
from tritonclient.utils import InferenceServerException


class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


# Callback function used for async_stream_infer()
def completion_callback(user_data, result, error):
    # passing error raise and handling out
    user_data._completed_requests.put((result, error))


class TritonInference:
    def __init__(self, model_name="", model_version=1, url='localhost:8000', protocol='http', async_set=False,
                 streaming=False, batch_size=1, classes=1, scaling='NONE', verbose=False):

        # Enable verbose output
        self.verbose = verbose
        # Use asynchronous inference API
        self.async_set = async_set
        # Use streaming inference API. The flag is only available with gRPC protocol.
        self.streaming = streaming
        # Name of model
        self.model_name = model_name
        # Version of model. Default is to use latest version.
        self.model_version = str(model_version)
        # Batch size. Default is 1.
        self.batch_size = batch_size
        # Number of class results to report. Default is 1.
        self.classes = classes
        # Type of scaling to apply to image pixels. Default is NONE.
        self.scaling = scaling
        # Inference server URL. Default is localhost:8000.
        self.url = url
        # 'Protocol (http/grpc) used to communicate with the inference service. Default is http.
        self.protocol = protocol
        # Input image / Input folder.
        # self.image_filename = None

        if self.streaming and self.protocol != "grpc":
            raise Exception("Streaming is only allowed with gRPC protocol")

        try:
            if self.protocol == "grpc":
                # Create gRPC client for communicating with the server
                self.triton_client = grpcclient.InferenceServerClient(
                    url=self.url, verbose=self.verbose)
            else:
                # Specify large enough concurrency to handle the
                # the number of requests.
                concurrency = 20 if self.async_set else 1
                self.triton_client = httpclient.InferenceServerClient(
                    url=self.url, verbose=self.verbose, concurrency=concurrency)
        except Exception as e:
            print("client creation failed: " + str(e))
            sys.exit(1)

        # Make sure the model matches our requirements, and get some
        # properties of the model that we need for preprocessing
        try:
            self.model_metadata = self.triton_client.get_model_metadata(
                model_name=self.model_name, model_version=self.model_version)
        except InferenceServerException as e:
            print("failed to retrieve the metadata: " + str(e))
            sys.exit(1)

        try:
            self.model_config = self.triton_client.get_model_config(
                model_name=self.model_name, model_version=self.model_version)
        except InferenceServerException as e:
            print("failed to retrieve the config: " + str(e))
            sys.exit(1)

        if self.protocol == "grpc":
            max_batch_size, input_name, output_name, c, h, w, format, dtype = self.parse_model_grpc()
        else:
            max_batch_size, input_name, output_name, c, h, w, format, dtype = self.parse_model_http()

        self.max_batch_size = max_batch_size
        self.input_name = input_name
        self.output_name = output_name
        self.c = c
        self.h = h
        self.w = w
        self.format = format
        self.dtype = dtype

    def parse_model_grpc(self):
        """
        Check the configuration of a model to make sure it meets the
        requirements for an image classification network (as expected by
        this client)
        """
        if len(self.model_metadata.inputs) != 1:
            raise Exception("expecting 1 input, got {}".format(
                len(self.model_metadata.inputs)))
        if len(self.model_metadata.outputs) != 1:
            raise Exception("expecting 1 output, got {}".format(
                len(self.model_metadata.outputs)))

        if len(self.model_config.input) != 1:
            raise Exception(
                "expecting 1 input in model configuration, got {}".format(
                    len(self.model_config.input)))

        input_metadata = self.model_metadata.inputs[0]
        input_config = self.model_config.input[0]
        output_metadata = self.model_metadata.outputs[0]

        if output_metadata.datatype != "FP32":
            raise Exception("expecting output datatype to be FP32, model '" +
                            self.model_metadata.name + "' output type is " +
                            output_metadata.datatype)

        # Output is expected to be a vector. But allow any number of
        # dimensions as long as all but 1 is size 1 (e.g. { 10 }, { 1, 10
        # }, { 10, 1, 1 } are all ok). Ignore the batch dimension if there
        # is one.
        output_batch_dim = (self.model_config.max_batch_size > 0)
        non_one_cnt = 0
        for dim in output_metadata.shape:
            if output_batch_dim:
                output_batch_dim = False
            elif dim > 1:
                non_one_cnt += 1
                if non_one_cnt > 1:
                    raise Exception("expecting model output to be a vector")

        # Model input must have 3 dims, either CHW or HWC (not counting
        # the batch dimension), either CHW or HWC
        input_batch_dim = (self.model_config.max_batch_size > 0)
        expected_input_dims = 3 + (1 if input_batch_dim else 0)
        if len(input_metadata.shape) != expected_input_dims:
            raise Exception(
                "expecting input to have {} dimensions, model '{}' input has {}".
                    format(expected_input_dims, self.model_metadata.name,
                           len(input_metadata.shape)))

        if ((input_config.format != mc.ModelInput.FORMAT_NCHW) and
                (input_config.format != mc.ModelInput.FORMAT_NHWC)):
            raise Exception("unexpected input format " +
                            mc.ModelInput.Format.Name(input_config.format) +
                            ", expecting " +
                            mc.ModelInput.Format.Name(mc.ModelInput.FORMAT_NCHW) +
                            " or " +
                            mc.ModelInput.Format.Name(mc.ModelInput.FORMAT_NHWC))

        if input_config.format == mc.ModelInput.FORMAT_NHWC:
            h = input_metadata.shape[1 if input_batch_dim else 0]
            w = input_metadata.shape[2 if input_batch_dim else 1]
            c = input_metadata.shape[3 if input_batch_dim else 2]
        else:
            c = input_metadata.shape[1 if input_batch_dim else 0]
            h = input_metadata.shape[2 if input_batch_dim else 1]
            w = input_metadata.shape[3 if input_batch_dim else 2]

        return (self.model_config.max_batch_size, input_metadata.name,
                output_metadata.name, c, h, w, input_config.format,
                input_metadata.datatype)

    def parse_model_http(self):
        """
        Check the configuration of a model to make sure it meets the
        requirements for an image classification network (as expected by
        this client)
        """
        if len(self.model_metadata['inputs']) != 1:
            raise Exception("expecting 1 input, got {}".format(
                len(self.model_metadata['inputs'])))
        if len(self.model_metadata['outputs']) != 1:
            raise Exception("expecting 1 output, got {}".format(
                len(self.model_metadata['outputs'])))

        if len(self.model_config['input']) != 1:
            raise Exception(
                "expecting 1 input in model configuration, got {}".format(
                    len(self.model_config['input'])))

        input_metadata = self.model_metadata['inputs'][0]
        input_config = self.model_config['input'][0]
        output_metadata = self.model_metadata['outputs'][0]

        max_batch_size = 0
        if 'max_batch_size' in self.model_config:
            max_batch_size = self.model_config['max_batch_size']

        if output_metadata['datatype'] != "FP32":
            raise Exception("expecting output datatype to be FP32, model '" +
                            self.model_metadata['name'] + "' output type is " +
                            output_metadata['datatype'])

        # Output is expected to be a vector. But allow any number of
        # dimensions as long as all but 1 is size 1 (e.g. { 10 }, { 1, 10
        # }, { 10, 1, 1 } are all ok). Ignore the batch dimension if there
        # is one.
        output_batch_dim = (max_batch_size > 0)
        non_one_cnt = 0
        for dim in output_metadata['shape']:
            if output_batch_dim:
                output_batch_dim = False
            elif dim > 1:
                non_one_cnt += 1
                if non_one_cnt > 1:
                    raise Exception("expecting model output to be a vector")

        # Model input must have 3 dims (not counting the batch dimension),
        # either CHW or HWC
        input_batch_dim = (max_batch_size > 0)
        expected_input_dims = 3 + (1 if input_batch_dim else 0)
        if len(input_metadata['shape']) != expected_input_dims:
            raise Exception(
                "expecting input to have {} dimensions, model '{}' input has {}".
                    format(expected_input_dims, self.model_metadata['name'],
                           len(input_metadata['shape'])))

        if ((input_config['format'] != "FORMAT_NCHW") and
                (input_config['format'] != "FORMAT_NHWC")):
            raise Exception("unexpected input format " + input_config['format'] +
                            ", expecting FORMAT_NCHW or FORMAT_NHWC")

        if input_config['format'] == "FORMAT_NHWC":
            h = input_metadata['shape'][1 if input_batch_dim else 0]
            w = input_metadata['shape'][2 if input_batch_dim else 1]
            c = input_metadata['shape'][3 if input_batch_dim else 2]
        else:
            c = input_metadata['shape'][1 if input_batch_dim else 0]
            h = input_metadata['shape'][2 if input_batch_dim else 1]
            w = input_metadata['shape'][3 if input_batch_dim else 2]

        return (max_batch_size, input_metadata['name'], output_metadata['name'], c,
                h, w, input_config['format'], input_metadata['datatype'])

    def preprocess(self, img):
        """
        Pre-process an image to meet the size, type and format
        requirements specified by the parameters.
        """
        # np.set_printoptions(threshold='nan')

        if self.c == 1:
            sample_img = img.convert('L')
        else:
            sample_img = img.convert('RGB')

        resized_img = sample_img.resize((self.w, self.h), Image.BILINEAR)
        resized = np.array(resized_img)
        if resized.ndim == 2:
            resized = resized[:, :, np.newaxis]

        npdtype = triton_to_np_dtype(self.dtype)
        typed = resized.astype(npdtype)

        if self.scaling == 'INCEPTION':
            scaled = (typed / 127.5) - 1
        elif self.scaling == 'VGG':
            if self.c == 1:
                scaled = typed - np.asarray((128,), dtype=npdtype)
            else:
                scaled = typed - np.asarray((123, 117, 104), dtype=npdtype)
        else:
            scaled = typed

        # Swap to CHW if necessary
        if self.protocol == "grpc":
            if self.format == mc.ModelInput.FORMAT_NCHW:
                ordered = np.transpose(scaled, (2, 0, 1))
            else:
                ordered = scaled
        else:
            if self.format == "FORMAT_NCHW":
                ordered = np.transpose(scaled, (2, 0, 1))
            else:
                ordered = scaled

        # Channels are in RGB order. Currently model configuration data
        # doesn't provide any information as to other channel orderings
        # (like BGR) so we just assume RGB.
        return ordered

    def postprocess(self, results, output_name):
        """
        Post-process results to show classifications.
        """

        output_array = results.as_numpy(output_name)
        if len(output_array) != self.batch_size:
            raise Exception("expected {} results, got {}".format(
                self.batch_size, len(output_array)))

        # Include special handling for non-batching models
        for results in output_array:
            if not self.max_batch_size > 0:
                results = [results]
            for result in results:
                if output_array.dtype.type == np.bytes_:
                    cls = "".join(chr(x) for x in result).split(':')
                else:
                    cls = result.split(':')
                print("    {} ({}) = {}".format(cls[0], cls[1], cls[2]))

    def request_generator(self, batched_image_data):
        # Set the input data
        inputs = []
        if self.protocol == "grpc":
            inputs.append(
                grpcclient.InferInput(self.input_name, batched_image_data.shape, self.dtype))
            inputs[0].set_data_from_numpy(batched_image_data)
        else:
            inputs.append(
                httpclient.InferInput(self.input_name, batched_image_data.shape, self.dtype))
            inputs[0].set_data_from_numpy(batched_image_data, binary_data=True)

        outputs = []
        if self.protocol == "grpc":
            outputs.append(
                grpcclient.InferRequestedOutput(self.output_name,
                                                class_count=self.classes))
        else:
            outputs.append(
                httpclient.InferRequestedOutput(self.output_name,
                                                binary_data=True,
                                                class_count=self.classes))

        yield inputs, outputs, self.model_name, self.model_version

    def __call__(self, image_filename):
        filenames = []
        if os.path.isdir(image_filename):
            filenames = [
                os.path.join(image_filename, f)
                for f in os.listdir(image_filename)
                if os.path.isfile(os.path.join(image_filename, f))
            ]
        else:
            filenames = [
                image_filename,
            ]

        filenames.sort()

        # Preprocess the images into input data according to model
        # requirements
        image_data = []
        for filename in filenames:
            img = Image.open(filename)
            image_data.append(
                self.preprocess(img))

        # Send requests of self.batch_size images. If the number of
        # images isn't an exact multiple of self.batch_size then just
        # start over with the first images until the batch is filled.
        # requests = []
        responses = []
        # result_filenames = []
        # request_ids = []
        image_idx = 0
        last_request = False
        user_data = UserData()

        # Holds the handles to the ongoing HTTP async requests.
        async_requests = []

        sent_count = 0

        if self.streaming:
            self.triton_client.start_stream(partial(completion_callback, user_data))

        while not last_request:
            input_filenames = []
            repeated_image_data = []

            for idx in range(self.batch_size):
                input_filenames.append(filenames[image_idx])
                repeated_image_data.append(image_data[image_idx])
                image_idx = (image_idx + 1) % len(image_data)
                if image_idx == 0:
                    last_request = True

            if self.max_batch_size > 0:
                batched_image_data = np.stack(repeated_image_data, axis=0)
            else:
                batched_image_data = repeated_image_data[0]

            # Send request
            try:
                for inputs, outputs, model_name, model_version in self.request_generator(
                        batched_image_data):
                    sent_count += 1
                    if self.streaming:
                        self.triton_client.async_stream_infer(
                            self.model_name,
                            inputs,
                            request_id=str(sent_count),
                            model_version=self.model_version,
                            outputs=outputs)
                    elif self.async_set:
                        if self.protocol == "grpc":
                            self.triton_client.async_infer(
                                self.model_name,
                                inputs,
                                partial(completion_callback, user_data),
                                request_id=str(sent_count),
                                model_version=self.model_version,
                                outputs=outputs)
                        else:
                            async_requests.append(
                                self.triton_client.async_infer(
                                    self.model_name,
                                    inputs,
                                    request_id=str(sent_count),
                                    model_version=self.model_version,
                                    outputs=outputs))
                    else:
                        responses.append(
                            self.triton_client.infer(self.model_name,
                                                     inputs,
                                                     request_id=str(sent_count),
                                                     model_version=self.model_version,
                                                     outputs=outputs))

            except InferenceServerException as e:
                print("inference failed: " + str(e))
                if self.streaming:
                    self.triton_client.stop_stream()
                sys.exit(1)

        if self.streaming:
            self.triton_client.stop_stream()

        if self.protocol == "grpc":
            if self.streaming or self.async_set:
                processed_count = 0
                while processed_count < sent_count:
                    (results, error) = user_data._completed_requests.get()
                    processed_count += 1
                    if error is not None:
                        print("inference failed: " + str(error))
                        sys.exit(1)
                    responses.append(results)
        else:
            if self.async_set:
                # Collect results from the ongoing async requests
                # for HTTP Async requests.
                for async_request in async_requests:
                    responses.append(async_request.get_result())

        for response in responses:
            if self.protocol == "grpc":
                this_id = response.get_response().id
            else:
                this_id = response.get_response()["id"]
            print("Request {}, batch size {}".format(this_id, self.batch_size))
            self.postprocess(response, self.output_name)

        print("PASS")


if __name__ == "__main__":
    model_name = "inception_graphdef"
    model_version = 1
    url = 'localhost:8000'
    protocol = 'http'
    async_set = False
    streaming = False
    batch_size = 1
    classes = 1
    scaling = 'INCEPTION'
    verbose = False
    image_filename = "../server/qa/images/mug.jpg"
    inference = TritonInference(model_name=model_name, model_version=model_version, url=url, protocol=protocol,
                                async_set=async_set, streaming=streaming, batch_size=batch_size, classes=classes,
                                scaling=scaling, verbose=verbose)
    inference(image_filename)
