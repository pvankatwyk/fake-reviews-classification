import pandas as pd
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

encoded_label_dict = {"CG" : 0, "OR" : 1}
def encode_label(x):
    return encoded_label_dict.get(x,-1)

df = pd.read_csv("../data/fake_reviews_dataset.csv")
df["target"] = df["label"].apply(lambda x: encode_label(x))

model_name = "roberta-base"
MAX_LEN = 256
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
EPOCHS = 1
LEARNING_RATE = 1e-05

texts = list(df['text_'])
labels = list(df['target'])

tokenizer = AutoTokenizer.from_pretrained("roberta-base") #Tokenizer
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='tf') #Tokenized text

dataset=tf.data.Dataset.from_tensor_slices((dict(inputs), labels)) #Create a tensorflow dataset
#train test split, we use 10% of the data for validation
val_data_size=int(0.1*len(dataset))
val_ds=dataset.take(val_data_size).batch(TRAIN_BATCH_SIZE, drop_remainder=True)
train_ds=dataset.skip(val_data_size).batch(TRAIN_BATCH_SIZE, drop_remainder=True)

n_categories = 2

model = TFAutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=n_categories)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.metrics.SparseCategoricalAccuracy(),
             tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='Sparse_Top_3_Categorical_Accuracy')],
)

h = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)

model.save_weights('./saved_weights.h5')