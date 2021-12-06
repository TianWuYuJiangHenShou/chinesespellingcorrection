from tritonclient.utils import *
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient

import numpy as np

#model_name = "ner_preprocess"
# model_name = "ner_ensemble"
model_name = "plome_ensemble"


#with httpclient.InferenceServerClient("172.26.0.126:11200") as client:
with grpcclient.InferenceServerClient("172.16.120.18:8001") as client:

    data = ["买一份人寿保险，可以吗","车祸你赔险有哪些","桑业保险包含生育理赔嘛","老年人可以参保的包险有哪些","女性任娠可以通过商业保险报销吗","父母有什么合适的长期包险可以买"]
    n = len(data)
    input0_data = np.array(data).astype(np.object_)
    input0_data = input0_data.reshape((n,-1))
    # input0_data = input0_data[:3]
    #print(input0_data)
    print(input0_data.shape)

    #inputs = [
    #    httpclient.InferInput("TEXT", input0_data.shape,
    #                          np_to_triton_dtype(input0_data.dtype))
    #]

    inputs = [
        grpcclient.InferInput("TEXT", input0_data.shape,
                              np_to_triton_dtype(input0_data.dtype))
    ]

    inputs[0].set_data_from_numpy(input0_data)
    #inputs[1].set_data_from_numpy(input1_data)

    #outputs = [
    #    httpclient.InferRequestedOutput("entity_indexs")
    #]

    outputs = [
        grpcclient.InferRequestedOutput("RAW"),
        grpcclient.InferRequestedOutput("CSC")
    ]

    response = client.infer(model_name,
                            inputs,
                            request_id=str(1),
                            outputs=outputs)

    result = response.get_response()
    #中文乱码反序列化
    raw = [line.decode('utf-8') for line in response.as_numpy("RAW")]
    csc = [line.decode('utf-8') for line in response.as_numpy("CSC")]

    for r,c in zip(raw,csc):
        print(r + '\t' + c)
