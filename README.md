# Snowflake Arctic-Embed Models
<h1 align="center">Snowflake's Artic-embed</h1>
<h4 align="center">
   <p>
       <a href=#news>News</a> |
       <a href=#models>Models</a> |
       <a href=#usage>Usage</a>  |
       <a href="#evaluation">Evaluation</a> |
       <a href="#contact">Contact</a> |
       <a href="#faq">FAQ</a>
       <a href="#license">License</a> |
       <a href="#acknowledgement">Acknowledgement</a>
   <p>
</h4>


## News

07/00/2024: Release v1.5 of our `m`-sized model, [snowflake-arctic-embed-m-v1.5](#snowflake-arctic-embed-m-v15)

05/10/2024: Release the [technical report on Arctic Embed](https://arxiv.org/html/2405.05374v1)


04/16/2024: Release the ** snowflake-arctic-embed ** family of text embedding models. The releases are state-of-the-art for Retrieval quality at each of their representative size profiles. Technical Report is coming shortly. For more details, please refer to our Github: [Arctic-Embed](https://github.com/Snowflake-Labs/arctic-embed).


## Models


snowflake-arctic-embed is a suite of text embedding models that focuses on creating high-quality retrieval models optimized for performance.


The `snowflake-arctic-embedding` models achieve **state-of-the-art performance on the MTEB/BEIR leaderboard** for each of their size variants. Evaluation is performed using these [scripts](https://github.com/Snowflake-Labs/snowflake-arctic-embed/tree/main/src). As shown below, each class of model size achieves SOTA retrieval accuracy compared to other top models.


The models are trained by leveraging existing open-source text representation models, such as bert-base-uncased, and are trained in a multi-stage pipeline to optimize their retrieval performance. First, the models are trained with large batches of query-document pairs where negatives are derived in-batch—pretraining leverages about 400m samples of a mix of public datasets and proprietary web search data. Following pretraining models are further optimized with long training on a smaller dataset (about 1m samples) of triplets of query, positive document, and negative document derived from hard harmful mining. Mining of the negatives and data curation is crucial to retrieval accuracy. A detailed technical report will be available shortly.


| Name                                                                    | MTEB Retrieval Score (NDCG @ 10) | Parameters (Millions) | Embedding Dimension |
| ----------------------------------------------------------------------- | -------------------------------- | --------------------- | ------------------- |
| [snowflake-arctic-embed-xs](https://huggingface.co/Snowflake/snowflake-arctic-embed-xs/)     | 50.15                            | 22                    | 384                 |
| [snowflake-arctic-embed-s](https://huggingface.co/Snowflake/snowflake-arctic-embed-s/)      | 51.98                            | 33                    | 384                 |
| [snowflake-arctic-embed-m](https://huggingface.co/Snowflake/snowflake-arctic-embed-m/)      | 54.90                            | 110                   | 768                 |
| [snowflake-arctic-embed-m-long](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-long/) | 54.83                            | 137                   | 768                 |
| [snowflake-arctic-embed-l](https://huggingface.co/Snowflake/snowflake-arctic-embed-l/)      | 55.98                            | 335                   | 1024                |


Aside from being great open-source models, the largest model, [snowflake-arctic-embed-l](https://huggingface.co/Snowflake/snowflake-arctic-embed-l/), can serve as a natural replacement for closed-source embedding, as shown below.


| Model Name                                                         | MTEB Retrieval Score (NDCG @ 10) |
| ------------------------------------------------------------------ | -------------------------------- |
| [snowflake-arctic-embed-l](https://huggingface.co/Snowflake/snowflake-arctic-embed-l/) | 55.98                            |
| Google-gecko-text-embedding                                        | 55.7                             |
| text-embedding-3-large                                             | 55.44                            |
| Cohere-embed-english-v3.0                                          | 55.00                            |
| bge-large-en-v1.5                                                  | 54.29                            |


### [snowflake-arctic-embed-xs](https://huggingface.co/Snowflake/snowflake-arctic-embed-xs)


This tiny model packs quite the punch. Based on the [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model with only 22m parameters and 384 dimensions, this model should meet even the strictest latency/TCO budgets. Despite its size, its retrieval accuracy is closer to that of models with 100m paramers.


| Model Name                                                          | MTEB Retrieval Score (NDCG @ 10) |
| ------------------------------------------------------------------- | -------------------------------- |
| [snowflake-arctic-embed-xs](https://huggingface.co/Snowflake/snowflake-arctic-embed-xs/) | 50.15                            |
| GIST-all-MiniLM-L6-v2                                               | 45.12                            |
| gte-tiny                                                            | 44.92                            |
| all-MiniLM-L6-v2                                                    | 41.95                            |
| bge-micro-v2                                                        | 42.56                            |


### [snowflake-arctic-embed-s](https://huggingface.co/Snowflake/snowflake-arctic-embed-s)


Based on the [intfloat/e5-small-unsupervised](https://huggingface.co/intfloat/e5-small-unsupervised) model, this small model does not trade off retrieval accuracy for its small size. With only 33m parameters and 384 dimensions, this model should easily allow scaling to large datasets.


| Model Name                                                         | MTEB Retrieval Score (NDCG @ 10) |
| ------------------------------------------------------------------ | -------------------------------- |
| [snowflake-arctic-embed-s](https://huggingface.co/Snowflake/snowflake-arctic-embed-s/) | 51.98                            |
| bge-small-en-v1.5                                                  | 51.68                            |
| Cohere-embed-english-light-v3.0                                    | 51.34                            |
| text-embedding-3-small                                             | 51.08                            |
| e5-small-v2                                                        | 49.04                            |


### [snowflake-arctic-embed-m](https://huggingface.co/Snowflake/snowflake-arctic-embed-m/)


Based on the [intfloat/e5-base-unsupervised](https://huggingface.co/intfloat/e5-base-unsupervised) model, this medium model is the workhorse that provides the best retrieval performance without slowing down inference.


| Model Name                                                         | MTEB Retrieval Score (NDCG @ 10) |
| ------------------------------------------------------------------ | -------------------------------- |
| [snowflake-arctic-embed-m](https://huggingface.co/Snowflake/snowflake-arctic-embed-m/) | 54.90                            |
| bge-base-en-v1.5                                                   | 53.25                            |
| nomic-embed-text-v1.5                                              | 53.25                            |
| GIST-Embedding-v0                                                  | 52.31                            |
| gte-base                                                           | 52.31                            |

### [snowflake-arctic-embed-m-long](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-long/)


Based on the [nomic-ai/nomic-embed-text-v1-unsupervised](https://huggingface.co/nomic-ai/nomic-embed-text-v1-unsupervised) model, this long-context variant of our medium-sized model is perfect for workloads that can be constrained by the regular 512 token context of our other models. Without the use of RPE, this model supports up to 2048 tokens. With RPE, it can scale to 8192!


| Model Name                                                         | MTEB Retrieval Score (NDCG @ 10) |
| ------------------------------------------------------------------ | -------------------------------- |
| [snowflake-arctic-embed-m-long](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-long/) | 54.83                            |
| nomic-embed-text-v1.5                                              | 53.01                            |
| nomic-embed-text-v1                                                | 52.81                            |




### [snowflake-arctic-embed-l](https://huggingface.co/Snowflake/snowflake-arctic-embed-l/)


Based on the [intfloat/e5-large-unsupervised](https://huggingface.co/intfloat/e5-large-unsupervised) model, this large model is a direct drop-in for closed APIs and delivers the most accurate retrieval experience.


| Model Name                                                         | MTEB Retrieval Score (NDCG @ 10) |
| ------------------------------------------------------------------ | -------------------------------- |
| [snowflake-arctic-embed-l](https://huggingface.co/Snowflake/snowflake-arctic-embed-l/) | 55.98                            |
| UAE-Large-V1                                                       | 54.66                            |
| bge-large-en-v1.5                                                  | 54.29                            |
| mxbai-embed-large-v1                                               | 54.39                            |
| e5-Large-v2                                                        | 50.56                            |


### [snowflake-arctic-embed-m-v1.5](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v1.5)

Based on the [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) model, this variant of our medium model is trained with [Matryoshka Representation Learning (MRL) loss](https://arxiv.org/abs/2205.13147) to deliver exceptional retrieval performance even when vectors are truncated to 256 dimensions.


| Model Name                                                         | MTEB Retrieval Score (NDCG @ 10) |
| ------------------------------------------------------------------ | -------------------------------- |
| [snowflake-arctic-embed-m-v1.5](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v1.5) | 55.17                            |
| [snowflake-arctic-embed-m](https://huggingface.co/Snowflake/snowflake-arctic-embed-m/) | 54.90                            |


| Model                         | Model Parameters   | MTEB Retrieval Score at 256 Dimensions (fraction of arctic-embed-m-v1.5)   |
|:------------------------------|:-------------------|:---------------------------------------------------------------------------|
| Snowflake arctic-embed-m-v1.5 | 109M               | 54.2 (100%)                                                                |
| Google gecko                  | 1200M              | 52.4 (97%)                                                                 |
| OpenAI text-embedding-3-large | Not Published      | 51.7 (96%)                                                                 |
| Nomic nomic-embed-text-v1.5   | 138M               | 50.8 (94%)                                                                 |


Additionally, this model was designed to pair well with a corpus-independent scalar quantization scheme to achieve great performance even in as little as 128 bytes per vector (24x compression compared to 768 dimensional vectors stored in float32).

| Model Version   |   Dimensionality | Scalar Quantization   | Bytes Per Vector (fraction of baseline)   | MTEB Retrieval Score (fraction of baseline)   | Vectors Per GB (improvement over baseline)   |
|:----------------|-----------------:|:----------------------|:------------------------------------------|:----------------------------------------------|:---------------------------------------------|
| v1              |              768 | None (float32)        | 3072 (100%)                               | 54.9 (100%)                                   | 0.33M (1.0x)                                 |
| v1              |              768 | int8                  | 768 (25%)                                 | 54.9 (100%)                                   | 1.3M (4x)                                    |
| v1.5            |              768 | int8                  | 768 (25%)                                 | 55.1 (100%)                                   | 1.3M (4x)                                    |
| v1.5            |              256 | int8                  | 256 (8.3%)                                | 54.2 (99%)                                    | 3.9M (12x)                                   |
| v1.5            |              256 | int4                  | 128 (4.2%)                                | 53.7 (98%)                                    | 7.8M (24x)                                   |

Good uniform scalar quantization range (used in above eval): -0.18 to 0.18. For a detailed walkthrough of int4 quantization with `snowflake-arctic-embed-m-v1.5`, check out our [example notebook](compressed_embeddings_examples/score_arctic_embed_m_v1dot5_with_quantization.ipynb).

## Usage


### Using Sentence Transformers

You can use the sentence-transformers package to use an snowflake-arctic-embed model. Here we show how to use our latest model, `snowflake-arctic-embed-m-v1.5`.

```python
from sentence_transformers import SentenceTransformer

# Model constant.
MODEL_ID = "Snowflake//snowflake-arctic-embed-m-v1.5"

# Your queries and docs.
queries = ['what is snowflake?', 'Where can I get the best tacos?']
documents = ['The Data Cloud!', 'Mexico City of Course!']

# Load the model.
model = SentenceTransformer(
    MODEL_ID, model_kwargs=dict(add_pooling_layer=False),
)

# Generate text embeddings.
query_embeddings = model.encode(queries, prompt_name="query")
document_embeddings = model.encode(documents)

# Scores via dotproduct.
scores = query_embeddings @ document_embeddings.T

# Pretty-print the results.
for query, query_scores in zip(queries, scores):
    doc_score_pairs = list(zip(documents, query_scores))
    doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
    print(f'Query: "{query}"')
    for document, score in doc_score_pairs:
        print(f'Score: {score:.4f} | Document: "{document}"')
    print()

#### OUTPUT ####
# Query: "what is snowflake?"
# Score: 0.3521 | Document: "The Data Cloud!"
# Score: 0.2358 | Document: "Mexico City of Course!"

# Query: "Where can I get the best tacos?"
# Score: 0.3884 | Document: "Mexico City of Course!"
# Score: 0.2389 | Document: "The Data Cloud!"
#

#### Variation: Truncated Embeddings ####
query_embeddings_256 = normalize(query_embeddings[:, :256])
doument_embeddings_256 = normalize(doument_embeddings[:, :256])
scores_256 = query_embeddings_256 @ doument_embeddings_256.T

# Pretty-print the results.
for query, query_scores in zip(queries, scores_256):
    doc_score_pairs = sorted(zip(documents, query_scores), key=lambda x: x[1], reverse=True)
    print(f'Query: "{query}"')
    for document, score in doc_score_pairs:
        print(f'Score: {score:.4f} | Document: "{document}"')
    print()

#### OUTPUT ####
# Query: "what is snowflake?"
# Score: 0.3852 | Document: "The Data Cloud!"
# Score: 0.2721 | Document: "Mexico City of Course!"

# Query: "Where can I get the best tacos?"
# Score: 0.4337 | Document: "Mexico City of Course!"
# Score: 0.2886 | Document: "The Data Cloud!"
#
```

### Using Huggingface transformers


You can use the transformers package to use an snowflake-arctic-embed model, too. Here we show how to use our latest model, `snowflake-arctic-embed-m-v1.5`. For optimal retrieval quality, use the CLS token to embed each text portion and use the query prefix below (just on the query).


```python
import torch
from torch.nn.functional import normalize
from transformers import AutoModel, AutoTokenizer

# Model constants.
MODEL_ID = "Snowflake/snowflake-arctic-embed-m-v1.5"
QUERY_PREFIX = 'Represent this sentence for searching relevant passages: '

# Your queries and docs.
queries  = ['what is snowflake?', 'Where can I get the best tacos?']
documents = ['The Data Cloud!', 'Mexico City of Course!']

# Load the model and tokenizer.
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID, add_pooling_layer=False)
model.eval()

# Add query prefix and tokenize queries and docs.
queries_with_prefix = [f"{QUERY_PREFIX}{q}" for q in queries]
query_tokens = tokenizer(queries_with_prefix, padding=True, truncation=True, return_tensors='pt', max_length=512)
document_tokens =  tokenizer(documents, padding=True, truncation=True, return_tensors='pt', max_length=512)

# Use the model to generate text embeddings.
with torch.inference_mode():
    query_embeddings = model(**query_tokens)[0][:, 0]
    doument_embeddings = model(**document_tokens)[0][:, 0]

# Remember to normalize embeddings.
query_embeddings = normalize(query_embeddings)
doument_embeddings = normalize(doument_embeddings)

# Scores via dotproduct.
scores = query_embeddings @ document_embeddings.T

# Pretty-print the results.
for query, query_scores in zip(queries, scores):
    doc_score_pairs = list(zip(documents, query_scores))
    doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
    print(f'Query: "{query}"')
    for document, score in doc_score_pairs:
        print(f'Score: {score:.4f} | Document: "{document}"')
    print()

#### OUTPUT ####
# Query: "what is snowflake?"
# Score: 0.3521 | Document: "The Data Cloud!"
# Score: 0.2358 | Document: "Mexico City of Course!"

# Query: "Where can I get the best tacos?"
# Score: 0.3884 | Document: "Mexico City of Course!"
# Score: 0.2389 | Document: "The Data Cloud!"
#

#### Variation: Truncated Embeddings ####
query_embeddings_256 = normalize(query_embeddings[:, :256])
doument_embeddings_256 = normalize(doument_embeddings[:, :256])
scores_256 = query_embeddings_256 @ doument_embeddings_256.T

# Pretty-print the results.
for query, query_scores in zip(queries, scores_256):
    doc_score_pairs = sorted(zip(documents, query_scores), key=lambda x: x[1], reverse=True)
    print(f'Query: "{query}"')
    for document, score in doc_score_pairs:
        print(f'Score: {score:.4f} | Document: "{document}"')
    print()

#### OUTPUT ####
# Query: "what is snowflake?"
# Score: 0.3852 | Document: "The Data Cloud!"
# Score: 0.2721 | Document: "Mexico City of Course!"

# Query: "Where can I get the best tacos?"
# Score: 0.4337 | Document: "Mexico City of Course!"
# Score: 0.2886 | Document: "The Data Cloud!"
#
```

### Usage Note: Long Context Embedding With `m-long`

If you use the long context model with more than 2048 tokens, ensure that you initialize the model like below instead. This will use [RoPE](https://arxiv.org/abs/2104.09864) to allow up to 8192 tokens.


``` python
model = AutoModel.from_pretrained(
    'Snowflake/snowflake-arctic-embed-m-long',
    trust_remote_code=True,
    safe_serialization=True,
    rotary_scaling_factor=2
)
```

### Using Transformers.js

If you haven't already, you can install the [Transformers.js](https://huggingface.co/docs/transformers.js) JavaScript library from [NPM](https://www.npmjs.com/package/@xenova/transformers) by running:
```bash
npm i @xenova/transformers
```

You can then compute embeddings from arctic-embed models as follows (`m-long` variant shown):

```js
import { pipeline, dot } from '@xenova/transformers';

// Create feature extraction pipeline
const extractor = await pipeline('feature-extraction', 'Snowflake/snowflake-arctic-embed-m-long', {
    quantized: false, // Comment out this line to use the quantized version
});

// Generate sentence embeddings
const sentences = [
    'Represent this sentence for searching relevant passages: Where can I get the best tacos?',
    'The Data Cloud!',
    'Mexico City of Course!',
]
const output = await extractor(sentences, { normalize: true, pooling: 'cls' });

// Compute similarity scores
const [source_embeddings, ...document_embeddings ] = output.tolist();
const similarities = document_embeddings.map(x => dot(source_embeddings, x));
console.log(similarities); // [0.36740492125676116, 0.42407774292046635]
```


## FAQ


TBD


## Contact


Feel free to open an issue or pull request if you have any questions or suggestions about this project.
You also can email Daniel Campos(daniel.campos@snowflake.com).


## License


Arctic is licensed under the [Apache-2](https://www.apache.org/licenses/LICENSE-2.0). The released models can be used for commercial purposes free of charge.


## Acknowledgement


We want to thank the open-source community, which has provided the great building blocks upon which we could make our models.
We thank our modeling engineers, Danmei Xu, Luke Merrick, Gaurav Nuti, and Daniel Campos, for making these great models possible. 
We thank our leadership, Himabindu Pucha, Kelvin So, Vivek Raghunathan, and Sridhar Ramaswamy, for supporting this work. 
We also thank the open-source community for producing the great models we could build on top of and making these releases possible. 
Finally, we thank the researchers who created BEIR and MTEB benchmarks. 
It is largely thanks to their tireless work to define what better looks like that we could improve model performance. 
