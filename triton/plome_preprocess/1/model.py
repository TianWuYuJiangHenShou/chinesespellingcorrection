import tokenization
from pinyin_tool import PinyinTool
from collections import namedtuple
import collections
import triton_python_backend_utils as pb_utils
import tensorflow as tf
import json

InputFeatures = namedtuple('InputFeature', ['input_ids', 'input_mask', 'segment_ids', 'stroke_ids'])

class plome_preprocess():
    def __init__(self):
        super(plome_preprocess,self).__init__()

        self.tokenizer = tokenization.FullTokenizer(vocab_file='/utils/vocab.txt', do_lower_case=False)

    def convert_single_sentence(self,sentence,max_sen_len,pytool,sktool):
        sentence = list(sentence.strip().replace(' ',''))
        # Account for [CLS] and [SEP] with "- 2"
        if len(sentence) > max_sen_len - 2:
            tokens = sentence[0:(max_sen_len - 2)]
        else:
            tokens = sentence

        _tokens = []
        # _labels = []
        # _lmask = []
        segment_ids = []
        stroke_ids = []
        _tokens.append("[CLS]")
        # _lmask.append(0)
        # _labels.append(labels[0])
        segment_ids.append(0)
        stroke_ids.append(0)
        for token in tokens:
            _tokens.append(token)
            # _labels.append(label)
            # _lmask.append(1)
            segment_ids.append(pytool.get_pinyin_id(token))
            stroke_ids.append(sktool.get_pinyin_id(token))
        input_ids = self.tokenizer.convert_tokens_to_ids(_tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_sen_len:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            stroke_ids.append(0)
            # _labels.append(labels[0])
            # _lmask.append(0)

        assert len(input_ids) == max_sen_len
        assert len(input_mask) == max_sen_len
        assert len(segment_ids) == max_sen_len
        assert len(stroke_ids) == max_sen_len

        # label_ids = [label_map.get(l, label_map['UNK']) for l in _labels]

        feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            stroke_ids=stroke_ids
        )
        return feature

    def decode_record(self,record):
        """Decodes a record to a TensorFlow example."""
        name_to_features = {
            "input_ids": tf.io.FixedLenFeature([180], tf.int64),
            "input_mask": tf.io.FixedLenFeature([180], tf.int64),
            "segment_ids": tf.io.FixedLenFeature([180], tf.int64),
            "stroke_ids": tf.io.FixedLenFeature([180], tf.int64)
        }


        example = tf.io.parse_single_example(serialized=record, features=name_to_features)

        #int64 to int64
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.cast(t, dtype=tf.int64)
            example[name] = t
        input_ids = example['input_ids']
        input_mask = example['input_mask']
        segment_ids = example['segment_ids']
        stroke_ids = example['stroke_ids']
        # label_ids = example['label_ids']
        # lmask = example['lmask']
        # py_labels = tf.py_func(_get_py_seq, [label_ids], [tf.int64])

        return input_ids, input_mask, segment_ids, stroke_ids

    def data_process(self,sentences,max_sen_len,batch_size):

        # label_list = {}
        # for key in self.tokenizer.vocab:
        #     label_list[self.tokenizer.vocab[key]] = key

        py_dict_path = '/utils/pinyin_data/zi_py.txt'
        py_vocab_path = '/utils/pinyin_data/py_vocab.txt'
        sk_dict_path = '/utils/stroke_data/zi_sk.txt'
        sk_vocab_path = '/utils/stroke_data/sk_vocab.txt'

        pytool = PinyinTool(py_dict_path=py_dict_path, py_vocab_path=py_vocab_path, py_or_sk='py')
        sktool = PinyinTool(py_dict_path=sk_dict_path, py_vocab_path=sk_vocab_path, py_or_sk='sk')

        # py_label_list = {v: k for k, v in pytool.vocab.items()}

        tokenid_pyid = {}
        tokenid_skid = {}
        for key in self.tokenizer.vocab:
            tokenid_pyid[self.tokenizer.vocab[key]] = pytool.get_pinyin_id(key)
            tokenid_skid[self.tokenizer.vocab[key]] = sktool.get_pinyin_id(key)

        ###DataProcessor init 结束
        # input_ids,input_masks,segment_ids,stroke_ids = [],[],[],[]
        features_ = []
        #print('****'*10)
        #print(sentences[:10])
        for sentence in sentences:
            feature = self.convert_single_sentence(sentence,max_sen_len,pytool,sktool)
            create_int_feature = lambda values: tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            features = collections.OrderedDict()
            features["input_ids"] = create_int_feature(feature.input_ids)
            features["input_mask"] = create_int_feature(feature.input_mask)
            features["segment_ids"] = create_int_feature(feature.segment_ids)
            features["stroke_ids"] = create_int_feature(feature.stroke_ids)

            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            features_.append(tf_example.SerializeToString())
        dataset = tf.data.Dataset.from_tensor_slices(features_)
        dataset = dataset.map(self.decode_record, num_parallel_calls=10)
        dataset = dataset.batch(batch_size).prefetch(16)
        dataset = list(dataset.as_numpy_iterator())
        input_ids, input_masks, segment_ids, stroke_ids = dataset[0]
        return input_ids, input_masks, segment_ids, stroke_ids

# data_loader = plome_preprocess()
# data = ["买一份人寿保险，可以吗","车祸你赔险有哪些","桑业保险包含生育理赔嘛","老年人可以参保的包险有哪些","女性任娠可以通过商业保险报销吗","父母有什么合适的长期包险可以买"]
# input_ids, input_masks, segment_ids, stroke_ids = data_loader.data_process(data,180,16)

# print(input_ids, input_masks, segment_ids, stroke_ids)
# print(type(input_ids),type(input_masks))
# print(dataset)

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
            model_config, "INPUT_IDS")
        output1_config = pb_utils.get_output_config_by_name(
            model_config, "INPUT_MASKS")
        output2_config = pb_utils.get_output_config_by_name(
            model_config, "SEGMENT_IDS")
        output3_config = pb_utils.get_output_config_by_name(
            model_config, "STROKE_IDS")

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])
        self.output1_dtype = pb_utils.triton_string_to_numpy(
            output1_config['data_type'])
        self.output2_dtype = pb_utils.triton_string_to_numpy(
            output2_config['data_type'])
        self.output3_dtype = pb_utils.triton_string_to_numpy(
            output3_config['data_type'])

        # Instantiate the PyTorch model
        self.data_loader = plome_preprocess()

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
        output2_dtype = self.output2_dtype
        output3_dtype = self.output3_dtype

        responses = []
        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "RAW_TEXT")
            #out_0, out_1 = self.add_sub_model(in_0.as_numpy(), in_1.as_numpy())
            inputs = []
            for instance in  in_0.as_numpy():
                sen = instance[0].decode('utf-8')
                #inputs.append(str(instance[0]))
                inputs.append(sen)
            out_0 ,out_1,out_2,out_3 = self.data_loader.data_process(inputs,max_sen_len= 180,batch_size = 16)
            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            out_tensor_0 = pb_utils.Tensor("INPUT_IDS",
                                           out_0.astype(output0_dtype))
            out_tensor_1 = pb_utils.Tensor("INPUT_MASKS",
                                           out_1.astype(output1_dtype))
            out_tensor_2 = pb_utils.Tensor("SEGMENT_IDS",
                                           out_2.astype(output2_dtype))
            out_tensor_3 = pb_utils.Tensor("STROKE_IDS",
                                           out_3.astype(output3_dtype))

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occured"))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0,out_tensor_1,out_tensor_2,out_tensor_3])

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
