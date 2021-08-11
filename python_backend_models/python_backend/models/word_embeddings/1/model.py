
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.

import triton_python_backend_utils as pb_utils
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device
import torch
from tqdm.autonotebook import trange
import time


def encode(model, sentences,
           batch_size: int = 32,
           show_progress_bar: bool = None,
           output_value: str = 'sentence_embedding',
           convert_to_numpy: bool = True,
           convert_to_tensor: bool = False,
           device: str = None,
           normalize_embeddings: bool = False):
    model.eval()
    if show_progress_bar is None:
        show_progress_bar = False

    if convert_to_tensor:
        convert_to_numpy = False

    if output_value == 'token_embeddings':
        convert_to_tensor = False
        convert_to_numpy = False

    input_was_string = False
    if isinstance(sentences, str) or not hasattr(sentences,
                                                 '__len__'):  # Cast an individual sentence to a list with length 1
        sentences = [sentences]
        input_was_string = True

    if device is None:
        device = model._target_device

    model.to(device)

    all_embeddings = []
    length_sorted_idx = np.argsort([-model._text_length(sen) for sen in sentences])
    sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
    tokenize_time = []
    model_time = []

    for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
        sentences_batch = sentences_sorted[start_index:start_index + batch_size]

        t1 = time.time()
        features = model.tokenize(sentences_batch)
        t2 = time.time()
        tokenize_time.append(t2-t1)
        features = batch_to_device(features, device)

        with torch.no_grad():
            out_features = model.forward(features)

            if output_value == 'token_embeddings':
                embeddings = []
                for token_emb, attention in zip(out_features[output_value], out_features['attention_mask']):
                    last_mask_id = len(attention) - 1
                    while last_mask_id > 0 and attention[last_mask_id].item() == 0:
                        last_mask_id -= 1

                    embeddings.append(token_emb[0:last_mask_id + 1])
            else:  # Sentence embeddings
                embeddings = out_features[output_value]
                embeddings = embeddings.detach()
                if normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                if convert_to_numpy:
                    embeddings = embeddings.cpu()

            all_embeddings.extend(embeddings)

        t3 = time.time()
        model_time.append(t3 - t2)

    all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

    if convert_to_tensor:
        all_embeddings = torch.stack(all_embeddings)
    elif convert_to_numpy:
        all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

    if input_was_string:
        all_embeddings = all_embeddings[0]

    return all_embeddings, (sum(tokenize_time)*1000), (sum(model_time)*1000)


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
        self.model = SentenceTransformer('paraphrase-mpnet-base-v2')

    def run_model(self, inp):
        inp = [v.decode("utf-8") for v in inp.as_numpy()]
        sentence_embeddings = encode(self.model, inp)
        return sentence_embeddings

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
        responses = []
        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for idx, request in enumerate(requests):
            # Get INPUT0
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            t1 = time.time()
            emb, token_time, model_time = self.run_model(in_0)
            t2 = time.time()
            ms_elapsed = (t2 - t1) * 1000
            out_tensor_0 = pb_utils.Tensor("OUTPUT0",
                                           emb)
            out_tensor_1 = pb_utils.Tensor("TIME_ELAPSED", np.array(ms_elapsed))
            out_tensor_2 = pb_utils.Tensor("TIME_TOKENIZE", np.array(token_time))
            out_tensor_3 = pb_utils.Tensor("TIME_MODEL", np.array(model_time))
            # Create InferenceResponse.
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0, out_tensor_1, out_tensor_2, out_tensor_3])
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