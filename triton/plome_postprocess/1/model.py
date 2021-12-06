# import tokenization
# from pinyin_tool import PinyinTool
from collections import namedtuple
import collections
import triton_python_backend_utils as pb_utils
import tensorflow as tf
import json
import tokenization
import numpy as np

class plome_postprocess():
    def __init__(self):
        super(plome_postprocess,self).__init__()
        self.label_list = {}
        tokenizer = tokenization.FullTokenizer(vocab_file='/utils/vocab.txt', do_lower_case=False)
        for key in tokenizer.vocab:
            self.label_list[tokenizer.vocab[key]] = key

    def csc_postprepare(self,inputs,fusion_preds):

        assert len(inputs) == len(fusion_preds)
        raw,csc,tmp1,tmp2 = [],[],[],[]

        for i,(input,pred) in enumerate(zip(inputs,fusion_preds)):
            if input == '[CLS]':
                if tmp1 == []:
                    continue
                else:
                    raw.append(tmp1)
                    csc.append(tmp2)
                    tmp1,tmp2 = [],[]
            else:
                tmp1.append(input)
                tmp2.append(pred)

            if i == len(inputs) - 1:
                raw.append(tmp1)
                csc.append(tmp2)
        return raw,csc

    def logits_convert(self,input_ids,gold_masks,fusion_probs):
        #print(input_ids.as_numpy().shape,gold_masks.as_numpy().shape,fusion_probs.as_numpy().shape)
        input_ids,gold_masks,fusion_probs = input_ids.as_numpy(),gold_masks.as_numpy(),fusion_probs.as_numpy()
        all_inputs,all_fusino_preds = [],[]
        fusion_probs = np.argmax(fusion_probs, axis=2)
        nums,max_sen_len = gold_masks.shape[0],gold_masks.shape[1]
        for k in range(nums):
            for j in range(max_sen_len):
                if gold_masks[k][j] == 0: continue
                all_inputs.append(input_ids[k][j])
                all_fusino_preds.append(fusion_probs[k][j])

        all_inputs = [self.label_list[k] for k in all_inputs]
        all_fusino_preds = [self.label_list[k] for k in all_fusino_preds]
        raw,csc = self.csc_postprepare(all_inputs,all_fusino_preds)
        raw = [''.join(line) for line in raw]
        csc = [''.join(line) for line in csc]
        return np.array(raw),np.array(csc)

class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args['model_config'])

        # Get OUTPUT configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "RAW_SENTENCE")
        output1_config = pb_utils.get_output_config_by_name(
            model_config, "CSC_SENTENCE")

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])
        self.output1_dtype = pb_utils.triton_string_to_numpy(
            output1_config['data_type'])

        # Instantiate the PyTorch model
        self.logits_convert = plome_postprocess()

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        print('Starting execute')
        output0_dtype = self.output0_dtype
        output1_dtype = self.output1_dtype

        responses = []
        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT_SENTENCE")
            in_1 = pb_utils.get_input_tensor_by_name(request, "CSC_MASKS")
            in_2 = pb_utils.get_input_tensor_by_name(request, "CONFUSION_PREDS")
            #out_0, out_1 = self.add_sub_model(in_0.as_numpy(), in_1.as_numpy())
            out_0 ,out_1 = self.logits_convert.logits_convert(in_0,in_1,in_2)
            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            out_tensor_0 = pb_utils.Tensor("RAW_SENTENCE",
                                           out_0.astype(output0_dtype))
            out_tensor_1 = pb_utils.Tensor("CSC_SENTENCE",
                                           out_1.astype(output1_dtype))

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occured"))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0,out_tensor_1])

            #inference_response = pb_utils.InferenceResponse(
            #    output_tensors=[out_tensor_0,out_tensor_1,out_tensor_2])
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
