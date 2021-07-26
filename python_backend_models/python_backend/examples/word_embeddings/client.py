import tritonclient.http as httpclient
from sentence_transformers import util
import numpy as np

model_name = "word_embeddings"
example_input = ["This is a red cat with a hat.", "Have you seen my red cat?"]

def run_inference(inp_sentence):
    with httpclient.InferenceServerClient("localhost:8000") as client:
        input0_data = np.array([inp_sentence], dtype=object)
        inputs = [
            httpclient.InferInput("INPUT0", input0_data.shape, "BYTES"),
        ]
        inputs[0].set_data_from_numpy(input0_data)

        outputs = [
            httpclient.InferRequestedOutput("OUTPUT0", binary_data=True),
        ]

        response = client.infer(model_name,
                                inputs,
                                request_id=str(1),
                                outputs=outputs)

        result = response.get_response()
        np_response = response.as_numpy("OUTPUT0").astype(float)
        return np_response
        #print(np_response.shape)

embeddings = []
for sent in example_input:
    emb = run_inference(sent)
    embeddings.append(emb)

cos_sim = util.cos_sim(embeddings[0], embeddings[1])
print("Cosine-Similarity:", cos_sim)