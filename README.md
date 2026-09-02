# Support Ticket Intent Classifier

Live demo: https://support-ticket-intent-classifier.vercel.app/

Backend API: https://support-ticket-intent-classifier.onrender.com

This is an end-to-end NLP project I built to classify customer support messages into 27 different intents.

I did not want to build only a model and stop there. My goal with this project was to go through the full ML workflow: preparing the data, building a baseline model from scratch, evaluating its weaknesses, fine-tuning a pretrained Transformer, optimizing the model for production, building an API, and finally deploying the whole application.

The project compares two approaches:

- a vanilla RNN that I implemented and trained from scratch
- a fine-tuned DistilBERT Transformer

The difference between the two models became especially clear when I tested them on messages that were written differently from the training data.

## Why I built this

At first, I trained the RNN and got very high accuracy on a normal random test split.

That looked good, but I did not fully trust it.

The dataset contains many messages that follow similar patterns, so a random split can make the test set look very similar to the training set.

Because of that, I created a separate challenge set with 81 manually written customer messages using more natural wording.

For example:

```text
Where is my package?
```

The RNN often struggled when the wording was different from what it had seen during training.

The fine-tuned DistilBERT model generalized much better.

## Results

| Model                     | Human-written challenge accuracy |
| ------------------------- | -------------------------------: |
| Vanilla RNN               |                           56.79% |
| DistilBERT                |                           90.12% |
| Quantized DistilBERT ONNX |                           90.12% |

The original DistilBERT and the quantized ONNX version both classified 73 out of 81 challenge examples correctly.

This was important because I wanted to make sure that optimizing the model for deployment did not reduce its classification performance on my challenge set.

## The RNN baseline

I first implemented a simple recurrent neural network in PyTorch.

I intentionally did not use `nn.RNN`.

The recurrent computation is implemented manually using:

- an embedding layer
- input-to-hidden linear transformation
- hidden-to-hidden linear transformation
- `tanh`
- a final classification layer

The recurrent update is essentially:

```python
new_hidden = torch.tanh(
    input_to_hidden(x_t)
    + hidden_to_hidden(hidden)
)
```

This helped me understand what is actually happening inside a basic recurrent network instead of treating the RNN as a black box.

The model uses the final valid hidden state to classify the message into one of the 27 support intents.

## Moving to DistilBERT

After seeing the RNN struggle on the challenge set, I wanted to test whether a pretrained language model could generalize better.

I fine-tuned:

```text
distilbert-base-uncased
```

for sequence classification with 27 output classes.

Unlike the RNN, DistilBERT already has pretrained language representations learned from a large text corpus.

During fine-tuning, the complete model was updated on the support-ticket dataset.

Training configuration included:

```text
Optimizer: AdamW
Learning rate: 2e-5
Weight decay: 0.01
Scheduler: linear warmup + decay
Epochs: 4
Max sequence length: 128
```

The best validation checkpoint was saved based on validation loss.

The final model reached:

```text
Validation accuracy: 99.65%
Human-written challenge accuracy: 90.12%
```

I consider the challenge result more useful for understanding real generalization because the random validation split contains wording and patterns similar to the training data.

## Deployment problem

When I first deployed the backend, I ran into a real production problem.

The original DistilBERT model was around:

```text
268 MB
```

and serving it through PyTorch required much more memory than the free Render instance provided.

Render's free instance had:

```text
512 MB RAM
```

and the application crashed with an out-of-memory error.

Instead of upgrading the server, I decided to optimize the inference pipeline.

## ONNX

I exported the trained models from PyTorch to ONNX.

ONNX allows the trained computation graph to run using ONNX Runtime without requiring PyTorch in the production server.

The production architecture changed from:

```text
FastAPI
   |
PyTorch
   |
RNN + DistilBERT
```

to:

```text
FastAPI
   |
ONNX Runtime
   |
RNN ONNX + DistilBERT ONNX
```

I also exported the manually implemented RNN to ONNX so that PyTorch could be removed completely from the production environment.

The ONNX RNN produced effectively the same output as the PyTorch version.

For the test message:

```text
Where is my package?
```

PyTorch RNN confidence:

```text
0.561200857
```

ONNX RNN confidence:

```text
0.561200500
```

Both predicted the same class.

## INT8 quantization

Exporting DistilBERT to ONNX alone did not reduce the model size very much.

The ONNX model was still approximately:

```text
256 MB
```

So I applied dynamic INT8 quantization.

The idea is to represent many of the model weights using 8-bit integers instead of 32-bit floating-point values during inference.

This reduced the model from:

```text
256 MB
```

to:

```text
64 MB
```

approximately a 4x reduction.

After quantization, I ran the complete 81-example challenge set again.

The result remained:

```text
73 / 81
90.12%
```

So in this case I was able to significantly reduce the model size without losing challenge-set accuracy.

This allowed the backend to run successfully on the free 512 MB Render instance.

## Final architecture

```text
                    ┌──────────────────────┐
                    │    Next.js frontend  │
                    │       Vercel         │
                    └──────────┬───────────┘
                               │
                               │ HTTP
                               ▼
                    ┌──────────────────────┐
                    │     FastAPI API      │
                    │       Render         │
                    └──────────┬───────────┘
                               │
                         ONNX Runtime
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
        Vanilla RNN ONNX            DistilBERT ONNX
            ~868 KB                   INT8 ~64 MB
                 │                           │
                 └─────────────┬─────────────┘
                               │
                         27 support intents
```

## Application

The frontend allows the same customer message to be sent to both models.

For every message, the application displays:

- RNN prediction
- RNN confidence
- DistilBERT prediction
- DistilBERT confidence

This makes it possible to directly compare the behavior of a model trained from scratch with a pretrained Transformer.

Example:

```text
Input:
Where is my package?

Vanilla RNN:
recover_password

DistilBERT:
track_order
```

## API

### Health check

```http
GET /health
```

### Compare both models

```http
POST /api/v1/predict/compare
```

Example request:

```json
{
  "text": "Where is my package?"
}
```

Example response:

```json
{
  "rnn": {
    "label": "recover_password",
    "confidence": 0.5612
  },
  "distilbert": {
    "label": "track_order",
    "confidence": 0.9105
  }
}
```

### Predict with one model

```http
POST /api/v1/predict
```

The API supports selecting either:

```text
rnn
distilbert
```

## Dataset

The project uses the Bitext customer support dataset.

The prepared dataset contains approximately:

```text
24,635 examples
27 intent classes
```

The preprocessing pipeline includes:

- data validation
- train / validation / test splitting
- tokenization
- vocabulary creation for the RNN
- padding and batching
- label encoding

DistilBERT uses its pretrained tokenizer instead of the custom RNN vocabulary.

## Project structure

```text
backend/
    api/
    core/
    middleware/
    schemas/
    services/
    main.py

frontend/
    app/

src/
    data/
    inference/
    models/
    transformer/

scripts/
    training
    evaluation
    ONNX export
    quantization
    inference tests

artifacts/
    config/
    vocab/
    rnn/
    distilbert/

data/
    raw/
    splits/
    challenge/
```

## Technologies

### Machine Learning

- Python
- PyTorch
- Hugging Face Transformers
- DistilBERT
- ONNX
- ONNX Runtime
- INT8 quantization
- NumPy

### Backend

- FastAPI
- Pydantic
- Uvicorn

### Frontend

- Next.js
- TypeScript
- React

### Deployment

- Vercel
- Render
- Hugging Face Hub
- GitHub

## What I learned from this project

One of the biggest things I learned from this project is that a high test accuracy does not automatically mean that a model will generalize well.

The RNN looked extremely strong on a random split, but when I changed the wording of the messages, its weaknesses became much clearer.

I also learned that training the model is only one part of an ML system.

Deployment introduced completely different problems:

- model size
- RAM usage
- inference dependencies
- API design
- CORS
- model loading
- production environments

The first deployment of DistilBERT failed because the server ran out of memory.

Instead of only increasing the server resources, I exported the models to ONNX, quantized DistilBERT to INT8, removed PyTorch from the production serving stack, and reduced the Transformer model from roughly 256 MB to 64 MB.

That process was one of the most useful parts of this project because it connected model development with actual ML deployment.

## Run locally

Clone the repository and install the Python dependencies:

```bash
pip install -r requirements.txt
```

Start the backend:

```bash
python -m uvicorn backend.main:app --reload
```

The API will run at:

```text
http://127.0.0.1:8000
```

Then start the frontend:

```bash
cd frontend
npm install
npm run dev
```

The frontend will run at:

```text
http://localhost:3000
```

## Live demo

https://support-ticket-intent-classifier.vercel.app/

The backend is hosted on Render's free tier, so after a period of inactivity the first request may take longer while the service starts again.
