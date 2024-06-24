import os
os.environ['HF_DATASETS_CACHE'] = '/dev/shm'

from typing import Any

from datasets import Dataset, load_dataset
import jax
from jax import Array
import jax.numpy as jnp
from jax_smi import initialise_tracking
import optax
from torch.utils.data import DataLoader
from transformers import BatchEncoding, FlaxT5ForConditionalGeneration, T5Tokenizer
import wandb

initialise_tracking()

# add your wandb api key here
WANDB_API_KEY = ...
dataset = load_dataset('wmt14', 'fr-en')
train_data = dataset['train'].select(range(50000))
val_data = dataset['validation']
test_data = dataset['test']

# 'd_model': 512,
# 'num_heads': 8,
# 'd_kv': 64
model = FlaxT5ForConditionalGeneration.from_pretrained('t5-base')
tokenizer = T5Tokenizer.from_pretrained('t5-base')

source_lang = 'en'
target_lang = 'fr'
prefix_text = 'translate English to French: '

num_epochs = 3
learning_rate = 1e-5
batch_size = 128
max_length = 64
key = jax.random.key(42)

wandb.login(key=WANDB_API_KEY)
wandb.init(
    project='training-t5',
    config=dict(batch_size=batch_size, learning_rate=learning_rate, max_length = 64)
)
os.environ['WANDB_LOG_MODEL'] = 'true'
os.environ['WANDB_WATCH'] = 'true'

def preprocess_function(sentences: Dataset) -> BatchEncoding:
    inputs = [prefix_text + sentence[source_lang] for sentence in sentences['translation']]
    targets = [sentence[target_lang] for sentence in sentences['translation']]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=max_length, truncation=True, padding='max_length')

    labels = jnp.array(model_inputs['labels'])
    decoder_input_ids = jnp.zeros_like(labels)
    decoder_input_ids = decoder_input_ids.at[:, 1:].set(labels[:, :-1])
    decoder_input_ids = decoder_input_ids.at[:, 0].set(tokenizer.pad_token_id)
    model_inputs['decoder_input_ids'] = decoder_input_ids

    return model_inputs

train_data = train_data.map(preprocess_function, batched=True, remove_columns=['translation'])
val_data = val_data.map(preprocess_function, batched=True, remove_columns=['translation'])

def jnp_collate_fn(batch: Any) -> Any:
    batch = {k: jnp.array([d[k] for d in batch]) for k in batch[0]}
    batch = jax.tree_util.tree_map(jnp.array, batch)
    return batch

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=jnp_collate_fn)
val_loader = DataLoader(val_data, batch_size=batch_size, collate_fn=jnp_collate_fn)

optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(model.params)

def compute_loss(logits: Array, labels: Array) -> Array:
    one_hot_labels = jax.nn.one_hot(labels, num_classes=logits.shape[-1])
    loss = optax.softmax_cross_entropy(logits, one_hot_labels).mean()
    return loss

def loss_fn(params: Any, inputs: Array, labels: Array, decoder_input_ids: Array, dropout_key: Array) -> Array:
    outputs = model(input_ids=inputs, decoder_input_ids=decoder_input_ids, params=params, train=True, dropout_rng=dropout_key)
    logits = outputs.logits
    loss = compute_loss(logits, labels)
    return loss

@jax.jit
def train_step(params: Any, opt_state: Any, batch: Any, key: Array) -> Any:
    inputs, labels, decoder_input_ids = batch['input_ids'], batch['labels'], batch['decoder_input_ids']

    loss, grads = jax.value_and_grad(loss_fn)(params, inputs, labels, decoder_input_ids, key)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss

def eval_step(params: Any, batch: Any) -> Array:
    inputs, labels, decoder_input_ids = batch['input_ids'], batch['labels'], batch['decoder_input_ids']
    outputs = model(input_ids=inputs, decoder_input_ids=decoder_input_ids, params=params, train=False)
    logits = outputs.logits
    loss = compute_loss(logits, labels)
    return loss

for epoch in range(num_epochs):
    for batch in train_loader:
        key, subkey = jax.random.split(key)
        model.params, opt_state, train_loss = train_step(model.params, opt_state, batch, subkey)
        wandb.log({
            "train_loss": train_loss,
        })

    val_loss = 0.
    for batch in val_loader:
        val_loss += eval_step(model.params, batch)
    val_loss /= len(val_loader)

    print(f'Epoch {epoch + 1}, Train Loss: {train_loss}, Validation Loss: {val_loss}')

wandb.finish()
