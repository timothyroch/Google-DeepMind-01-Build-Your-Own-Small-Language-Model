import os

os.environ["KERAS_BACKEND"] = "jax"

import keras
import pandas as pd
import tensorflow as tf

from ai_foundations import training
from gdm_lab_1_4_prepare_the_dataset_for_training_a_slm import SimpleWordTokenizer


keras.utils.set_random_seed(812)


# Load the training data
africa_galore = pd.read_json(
    "https://storage.googleapis.com/"
    "dm-educational/assets/ai_foundations/africa_galore.json"
)

dataset = africa_galore["description"].values

print("Loaded dataset with", dataset.shape[0], "paragraphs.")


# Build the vocabulary and encode each paragraph
tokenizer = SimpleWordTokenizer(dataset)

encoded_tokens = []

for text in dataset:
    encoded_tokens.append(tokenizer.encode(text))

print("Vocabulary size:", tokenizer.vocabulary_size)


# Check paragraph lengths before choosing the padded length
shortest_paragraph_length = min(len(row) for row in encoded_tokens)
longest_paragraph_length = max(len(row) for row in encoded_tokens)

print("Shortest paragraph:", shortest_paragraph_length)
print("Longest paragraph:", longest_paragraph_length)


# Short paragraphs are padded and long ones are truncated
max_length = 130

padded_sequences = keras.utils.pad_sequences(
    encoded_tokens,
    maxlen=max_length,
    padding="post",
    truncating="post",
    value=tokenizer.pad_token_id,
)

print("Padded dataset shape:", padded_sequences.shape)


# The target is the same sequence shifted one token to the left
input_sequences = padded_sequences[:, :-1]
target_sequences = padded_sequences[:, 1:]

print("Input shape:", input_sequences.shape)
print("Target shape:", target_sequences.shape)

max_length = input_sequences.shape[1]


# Keep each input together with its corresponding target
tf_dataset = tf.data.Dataset.from_tensor_slices(
    (input_sequences, target_sequences)
)

# Shuffle paragraphs, not the tokens inside them
tf_dataset = tf_dataset.shuffle(
    buffer_size=len(input_sequences)
)


# Split the dataset into batches
batch_size = 32
batches = tf_dataset.batch(batch_size)

print("Number of batches:", len(list(batches)))


# Create the transformer used in the lab
model = training.create_model(
    max_length=max_length,
    vocabulary_size=tokenizer.vocabulary_size,
    learning_rate=1e-4,
)


# Generate from the same prompt every 10 epochs to see how training progresses
prompt = "Abeni,"
prompt_ids = tokenizer.encode(prompt)

text_gen_callback = training.TextGenerator(
    max_tokens=10,
    start_tokens=prompt_ids,
    tokenizer=tokenizer,
    print_every=10,
)


# Train the model
num_epochs = 200

history = model.fit(
    x=batches,
    epochs=num_epochs,
    verbose=2,
    callbacks=[text_gen_callback],
)
