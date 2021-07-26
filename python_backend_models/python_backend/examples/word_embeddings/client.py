
import tritonclient.http as httpclient
from sentence_transformers import util
import numpy as np

"example reference taken from: https://www.sbert.net/docs/quickstart.html#comparing-sentence-similarities"

model_name = "word_embeddings"
sentences = ['A man is eating food.',
          'A man is eating a piece of bread.',
          'The girl is carrying a baby.',
          'A man is riding a horse.',
          'A woman is playing violin.',
          'Two men pushed carts through the woods.',
          'A man is riding a white horse on an enclosed ground.',
          'A monkey is playing drums.',
          'Someone in a gorilla costume is playing a set of drums.'
          ]
def run_inference(inp_sentence):
    with httpclient.InferenceServerClient("localhost:8000") as client:
        input0_data = np.array(inp_sentence, dtype=object)
        print(input0_data.shape)
        inputs = [
            httpclient.InferInput("INPUT0", input0_data.shape, "BYTES"),
        ]
        inputs[0].set_data_from_numpy(input0_data)

        outputs = [
            httpclient.InferRequestedOutput("OUTPUT0", binary_data=True),
        ]

        response = client.infer(model_name,
                                inputs,
                                outputs=outputs)

        print(response.get_response())
        result = response.get_response()
        np_response = response.as_numpy("OUTPUT0").astype(float)
        return np_response
        #print(np_response.shape)

#Encode all sentences at once
embeddings = run_inference(sentences)
# embeddings shape
print(embeddings.shape)

#Compute cosine similarity between all pairs
cos_sim = util.cos_sim(embeddings, embeddings)

#Add all pairs to a list with their cosine similarity score
all_sentence_combinations = []
for i in range(len(cos_sim)-1):
    for j in range(i+1, len(cos_sim)):
        all_sentence_combinations.append([cos_sim[i][j], i, j])

#Sort list by the highest cosine similarity score
all_sentence_combinations = sorted(all_sentence_combinations, key=lambda x: x[0], reverse=True)

print("Top-5 most similar pairs:")
for score, i, j in all_sentence_combinations[0:5]:
    print("{} \t {} \t {:.4f}".format(sentences[i], sentences[j], cos_sim[i][j]))
