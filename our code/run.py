from model import RobertaModel
from transformers import RobertaTokenizer
import pandas as pd
import tensorflow as tf

model_name = "roberta-base"
MAX_LEN = 256
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
EPOCHS = 1
LEARNING_RATE = 1e-05

encoded_label_dict = {"CG": 0, "OR": 1}


def encode_label(x):
    return encoded_label_dict.get(x, -1)


df = pd.read_csv("../data/fake_reviews_dataset.csv")
df["target"] = df["label"].apply(lambda x: encode_label(x))

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
train_text = list(df['text_'])
train_label = list(df['target'])
train_encodings = tokenizer(train_text,
                            None,
                            add_special_tokens=True,
                            max_length=MAX_LEN,
                            padding='max_length',
                            return_token_type_ids=True,
                            truncation=True,
                            )
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_label
))

model = RobertaModel()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=['accuracy'])
# model.compile(optimizer=optimizer, loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])
model.fit(train_dataset.batch(TRAIN_BATCH_SIZE), epochs=EPOCHS, batch_size=1)
