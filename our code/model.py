import tensorflow as tf

class RobertaLayer(tf.keras.layers.Layer):
    def __init__(self, name):
        super(RobertaLayer, self).__init__(name)
        # query, key, value, dropout
        self.query = tf.keras.layers.Dense(768)
        self.value = tf.keras.layers.Dense(768)
        self.key = tf.keras.layers.Dense(768)
        self.embeddings = tf.keras.layers.Attention(
            dropout=0.1, name="attention")
        self.embeddings_out = tf.keras.Sequential([
            tf.keras.layers.Dense(768),
            tf.keras.layers.LayerNormalization(epsilon=1e-05),
            tf.keras.layers.Dropout(0.1)
        ], name="embeddings_out")
        self.intermediate = tf.keras.layers.Dense(3072)
        self.out = tf.keras.Sequential([
            tf.keras.layers.Dense(768),
            tf.keras.layers.LayerNormalization(epsilon=1e-05),
            tf.keras.layers.Dropout(0.1)
        ], name="out")

    def call(self, inputs):
        query = self.query(inputs)
        key = self.key(inputs)
        value = self.value(inputs)
        embeddings = self.embeddings(query, key, value)
        embeddings_out = self.embeddings_out(embeddings_out)
        intermediate = self.intermediate(embeddings)
        output = self.out(intermediate)
        return output


class RobertaModel(tf.keras.Model):
    def __init__(self):
        super(RobertaModel, self).__init__()
        self.embeddings = tf.keras.Sequential([
            # ignoring padding_idx=1 for now
            tf.keras.layers.Embedding(50265, 768, name="word_embeddings"),
            tf.keras.layers.Embedding(50265, 768, name="position_embeddings"),
            tf.keras.layers.Embedding(
                50265, 768, name="token_type_embeddings"),
            # pytorch says 768 and elementwise_affine=True
            tf.keras.layers.LayerNormalization(epsilon=1e-05),
            tf.keras.layers.Dropout(0.1)
        ], name="embeddings")

        self.encoder = tf.keras.Sequential([
            RobertaLayer(name="Roberta0"),
            RobertaLayer(name="Roberta1"),
            RobertaLayer(name="Roberta2"),
            RobertaLayer(name="Roberta3"),
            RobertaLayer(name="Roberta4"),
            RobertaLayer(name="Roberta5"),
            RobertaLayer(name="Roberta6"),
            RobertaLayer(name="Roberta7"),
            RobertaLayer(name="Roberta8"),
            RobertaLayer(name="Roberta9"),
            RobertaLayer(name="Roberta10"),
            RobertaLayer(name="Roberta11"),
        ], name="encoder")

        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(768),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(2)
        ], name="classifier")
    
    def call(self, x):
        embeddings = self.embeddings(x)
        encoded = self.encoder(embeddings)
        decision = self.classifier(encoded)
        return decision