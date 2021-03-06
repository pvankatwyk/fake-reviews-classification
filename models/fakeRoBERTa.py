"""pretrained.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PADxSJtC9fbOh86Oyk_crKwYZKeOBzor

# Fake Reviews Detection
See https://devpost.com/software/fake-reviews-classification for the project overview, group members, and goal.

### Setup
"""

import pandas as pd
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
import tqdm.notebook
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import matplotlib.pyplot as plt


"""### Data Processing"""

encoded_label_dict = {"CG": 0, "OR": 1}


def encode_label(x):
    return encoded_label_dict.get(x, -1)


df = pd.read_csv("/content/fake-reviews-classification/data/fake_reviews_dataset.csv")
df["target"] = df["label"].apply(lambda x: encode_label(x))

train, test = train_test_split(df, test_size=0.2, shuffle=True, stratify=None, random_state=2021)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

model_name = "roberta-base"
MAX_LEN = 256
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
EPOCHS = 1
LEARNING_RATE = 1e-05

texts = list(train['text_'])
labels = list(train['target'])

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
inputs = tokenizer(texts, max_length=MAX_LEN, padding='max_length', truncation=True, return_tensors='tf')

dataset = tf.data.Dataset.from_tensor_slices((dict(inputs), labels))
# Val dataset size: 10%
val_data_size = int(0.1 * len(dataset))
val_ds = dataset.take(val_data_size).batch(TRAIN_BATCH_SIZE, drop_remainder=True)
train_ds = dataset.skip(val_data_size).batch(VALID_BATCH_SIZE, drop_remainder=True)


class SaveBatchLogs(tf.keras.callbacks.Callback):
    def __init__(self, loss_list):
        super(SaveBatchLogs, self).__init__()
        self.loss_list = loss_list

    def on_train_batch_end(self, batch, logs=None):
        self.loss_list.append(logs)


model = TFAutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.metrics.SparseCategoricalAccuracy()],
)

batch_logs = []
h = model.fit(train_ds,
              validation_data=val_ds,
              epochs=EPOCHS,
              callbacks=SaveBatchLogs(batch_logs))

# {'loss': [0.13365936279296875],
#  'sparse_categorical_accuracy': [0.951381266117096],
#  'val_loss': [0.1406628042459488],
#  'val_sparse_categorical_accuracy': [0.9678217768669128]}

model.save_weights('./saved_weights.h5')

h.history

# WRITE LOGS TO TXT FILE
import json

textfile = open("logs.txt", "w")
for element in batch_logs:
    textfile.write(json.dumps(element))
textfile.close()

acc = []
loss = []
for log in batch_logs:
    loss.append(log['loss'])
    acc.append(log['sparse_categorical_accuracy'])


# list all data in history
print(h.history.keys())
# summarize history for accuracy
plt.plot(loss, label='loss')
plt.title('Model Loss (1 Epoch)')
plt.ylabel('loss')
plt.xlabel('batch')
plt.savefig(r'loss.png')
plt.show()
# summarize history for loss
plt.plot(acc, label='accuracy')
plt.title('Model Accuracy (1 Epoch)')
plt.ylabel('accuracy')
plt.xlabel('batch')
plt.savefig('accuracy.png')
plt.show()

# tf.saved_model.save(model, 'model')

def predict(query, model, tokenizer):
    tokens = tokenizer.encode(query, return_tensors="tf")
    all_tokens = len(tokens)
    mask = tf.ones_like(tokens)

    logits = model(tokens, attention_mask=mask)[0]
    probs = tf.nn.softmax(logits)
    fake, real = probs.numpy().tolist()[0]
    return real, fake


query = """I work in the wedding industry and have to work long days, on my feet, outside in the heat, and have to look professional. I've spent a ridiculous amount of money on high end dress shoes like Merrels and just have not been able to find a pair that are comfortable to wear all day. Both for my feet and my back. Enter the Sanuk yoga sling!!! These shoes are amazingly comfortable. Though, I will admit it took a few wears to get used to the feel of the yoga matte bottom. At first, it felt a little "sticky" to me, and the fabric part that goes through the toe area was a little thick and took some getting used to. I wore them for a few days before taking them out on a job and I can't get over how comfortable they are. Ii have been wearing these shoes now for 3 months, every work day and I am THRILLED. No more back pain, no more sore feet. I also wear these sometimes during my off time,mans every time I wear them, I get compliments on how cute and comfortable they look. The great thing about these shoes is the yoga matte bottom. It helps your feet grip to the shoe a bit, so your foot can just walk normally, without having to grip the shoe. You may not realize it, but with a lot of Sandals, your foot is having to work to keep the shoe on, changing the way you walk and stand and ultimately causing foot and back pain. Not with these! Also, the soft linen sits comfortably on your skin and breathes nicely in the heat. The only downside is the funky tan lines, which is why I am sure to alternate shoes on my days off, especially if I plan to be outside for most of the day. If it were not for that, I think these might be the only shoes I'd wear all summer. If you are looking for a reasonable priced, comfortable shoe that you can wear and walk in all day."""
real, fake = predict(query, model, tokenizer)
print(f"Real Probability: {real}\nFake Probability: {fake}")

query = """My old bet was wearing this to the Macy's in January.  This is the first one I've ever had.  I am a 32D, and the first pair I bought were just a little tight.  I'm a bit disappointed.  This is my second pair.  I'm looking forward to wearing them to the Macy's in the fall.  I like the way they look.Love these!These are my favorite.  I have a hard time finding jeans that fit me comfortably, but I have a hard time finding jeans that don't fit.  These jeans are super comfortable and have a great price point.  I have some great jeans to wear for work, but these are the only jeans that I wear for work or for my family.  I will be buying more!  I have a lot of compliments on them.I love these shoes. I love the color and the fit. They fit my body well and are comfortable. I have a wide foot and these fit me well.

I'm 5'4", 130lbs and these fit well. I would recommend them.I wear a size 11.5 in jeans and this fits perfect. I have a narrow foot and this fits perfect. It is very comfortable and fits great. I bought a small and it fit perfectly. I will order another size up.I bought these for my husband, he loves them and he loves them!This is the best pair of sunglasses for the price!  They are so comfortable and easy to use.  I wear them all the time and they don't hurt my feet.  I wear them everyday and my feet are so happy with them!"""
real, fake = predict(query, model, tokenizer)
print(f"Real Probability: {real}\nFake Probability: {fake}")

query = """I work in the wedding industry and have to work long days, on my feet, outside in the heat, and have to look professional. I've spent a ridiculous amount of money on high end dress shoes like Merrels and just have not been able to find a pair that are comfortable to wear all day. Both for my feet and my back. Enter the Sanuk yoga sling!!! These shoes are amazingly comfortable. Though, I will admit it took a few wears to get used to the feel of the yoga matte bottom. At first, it felt a little "sticky" to me, and the fabric part that goes through the toe area was a little thick and took some getting used to. I wore them for a few days before taking them out on a job and I can't get over how comfortable they are. Ii have been wearing these shoes now for 3 months, every work day and I am THRILLED. No more back pain, no more sore feet. I also wear these sometimes during my off time,mans every time I wear them, I get compliments on how cute and comfortable they look. The great thing about these shoes is the yoga matte bottom. It helps your feet grip to the shoe a bit, so your foot can just walk normally, without having to grip the shoe. You may not realize it, but with a lot of Sandals, your foot is having to work to keep the shoe on, changing the way you walk and stand and ultimately causing foot and back pain. Not with these! Also, the soft linen sits comfortably on your skin and breathes nicely in the heat. The only downside is the funky tan lines, which is why I am sure to alternate shoes on my days off, especially if I plan to be outside for most of the day. If it were not for that, I think these might be the only shoes I'd wear all summer. If you are looking for a reasonable priced, comfortable shoe that you can wear and walk in all day."""
real, fake = predict(query, model, tokenizer)
print(f"Real Probability: {real}\nFake Probability: {fake}")

# testing
test_text = test['text_']
test_labels = test['target']

preds, preds_probas = [], []
for i, text in enumerate(tqdm.tqdm(test_text)):
    pred = predict(text, model, tokenizer)[0]
    preds_probas.append(pred)
    if pred >= 0.5:
        preds.append(1)
    else:
        preds.append(0)

from sklearn.metrics import confusion_matrix

y_true = list(test_labels)
y_pred = preds
confusion_matrix(y_true, y_pred)

from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

acc = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)

print(f"Accuracy: {acc * 100}; Precision:{precision * 100}; Recall:{recall * 100}")

print(classification_report(y_true, y_pred, target_names=["CG", "OR"]))
