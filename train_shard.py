import os
os.environ['HF_DATASETS_CACHE'] = '/dev/shm'

from typing import Any

from datasets import Dataset, load_dataset
from einshard import einshard
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
model = FlaxT5ForConditionalGeneration.from_pretrained('t5-large')
tokenizer = T5Tokenizer.from_pretrained('t5-large')

source_lang = 'en'
target_lang = 'fr'
prefix_text = 'translate English to French: '

num_epochs = 3
learning_rate = 1e-5
batch_size = 128
max_length = 64
key = jax.random.key(42)
is_host0 = jax.process_index() == 0

if is_host0:
    wandb.login(key=WANDB_API_KEY)
    wandb.init(
        project='training-t5',
        config=dict(
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_length=max_length,
        ),
    )
    os.environ['WANDB_LOG_MODEL'] = 'true'
    os.environ['WANDB_WATCH'] = 'true'

def apply_einshard(k: str, v: Array) -> Array:
    if 'Attention' in k and 'kernel' in k:
        expr = 'm r -> m r*'
    elif 'wi' in k:
        expr = 'm f -> m f*'
    elif 'wo' in k:
        expr = 'm f -> m* f'
    else:
        expr = '... -> * ...'
    v = einshard(v, expr)
    return v

def shard_params(params: Any, parent_key: str = '') -> Any:
    if isinstance(params, dict):
        for k, v in params.items():
            full_key = f'{parent_key}.{k}' if parent_key else k
            if isinstance(v, (dict, list)):
                params[k] = shard_params(v, full_key)
            else:
                params[k] = apply_einshard(full_key, v)
    elif isinstance(params, list):
        for i in range(len(params)):
            full_key = f'{parent_key}[{i}]'
            if isinstance(params[i], (dict, list)):
                params[i] = shard_params(params[i], full_key)
            else:
                params[i] = apply_einshard(full_key, params[i])
    return params

model.params = shard_params(model.params)

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

        if is_host0:
            wandb.log({
                "train_loss": train_loss,
            })

    val_loss = 0.
    for batch in val_loader:
        val_loss += eval_step(model.params, batch)
    val_loss /= len(val_loader)

    print(f'Epoch {epoch + 1}, Train Loss: {train_loss}, Validation Loss: {val_loss}')

if is_host0:
    wandb.finish()
