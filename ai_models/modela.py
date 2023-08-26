import pandas as pd
from transformers import InputExample, InputFeatures
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures
from sklearn.model_selection import train_test_split
from sklearn import model_selection, datasets
from sklearn.tree import DecisionTreeClassifier
import joblib
import pickle
import os.path


def convert_data_to_examples(train, test, DATA_COLUMN, LABEL_COLUMN):
    train_InputExamples = train.apply(
        lambda x: InputExample(guid=None,  # Globally unique ID for bookkeeping, unused in this case
                               text_a=x[DATA_COLUMN],
                               text_b=None,
                               label=x[LABEL_COLUMN]), axis=1)

    validation_InputExamples = test.apply(
        lambda x: InputExample(guid=None,  # Globally unique ID for bookkeeping, unused in this case
                               text_a=x[DATA_COLUMN],
                               text_b=None,
                               label=x[LABEL_COLUMN]), axis=1)

    return train_InputExamples, validation_InputExamples

def convert_examples_to_tf_dataset(examples, tokenizer, max_length=128):
    features = [] # -> will hold InputFeatures to be converted later

    for e in examples:
        # Documentation is really strong for this method, so please take a look at it
        input_dict = tokenizer.encode_plus(
            e.text_a,
            add_special_tokens=True,
            max_length=max_length, # truncates if len(s) > max_length
            return_token_type_ids=True,
            return_attention_mask=True,
            pad_to_max_length=True, # pads to the right by default # CHECK THIS for pad_to_max_length
            truncation=True
        )

        input_ids, token_type_ids, attention_mask = (input_dict["input_ids"],
            input_dict["token_type_ids"], input_dict['attention_mask'])

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=e.label
            )
        )

    def gen():
        for f in features:
            yield (
                {
                    "input_ids": f.input_ids,
                    "attention_mask": f.attention_mask,
                    "token_type_ids": f.token_type_ids,
                },
                f.label,
            )

    return tf.data.Dataset.from_generator(
        gen,
        ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
        (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None]),
                "token_type_ids": tf.TensorShape([None]),
            },
            tf.TensorShape([]),
        ),
    )

def model1(text):

    if  not os.path.isfile('model_weights.index'):
        df=pd.read_csv("static/ratings40.csv",encoding = 'latin-1')
        df['txt']=df.Reviews
        df['target']=df.Ratings
        df=df.drop(['Ratings','Reviews'],axis=1)
        model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        df=df.dropna()
        x_train, x_test, y_train, y_test = train_test_split(df['txt'], df['target'], test_size=0.2)
        train=pd.concat([x_train,y_train],axis=1)
        test=pd.concat([x_test,y_test],axis=1)
        InputExample(guid=None,
             text_a = "Hello, world",
             text_b = None,
             label = 1)
        train_InputExamples, validation_InputExamples = convert_data_to_examples(train,test,
                                                                           'txt',
                                                                           'target')
        DATA_COLUMN = 'txt'
        LABEL_COLUMN = 'target'
        train_InputExamples, validation_InputExamples = convert_data_to_examples(train, test, DATA_COLUMN, LABEL_COLUMN)
        train_data = convert_examples_to_tf_dataset(list(train_InputExamples), tokenizer)
        train_data = train_data.shuffle(100).batch(32).repeat(2)
        validation_data = convert_examples_to_tf_dataset(list(validation_InputExamples), tokenizer)
        validation_data = validation_data.batch(32)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])
        model.fit(train_data, epochs=2, validation_data=validation_data)
        os.listdir('savedmodel')
        model.save_weights('savedmodel')
        model
        pred_sentences = [text]
        tf_batch = tokenizer(pred_sentences, max_length=128, padding=True, truncation=True, return_tensors='tf')
        tf_outputs = model(tf_batch)
        tf_predictions = tf.nn.softmax(tf_outputs[0], axis=-1)
        labels = ['Negative', 'Positive']
        label = tf.argmax(tf_predictions, axis=1)
        label = label.numpy()
        res = {}
        for i in range(len(pred_sentences)):
            print(pred_sentences[i], ": \n", labels[label[i]])
            res[pred_sentences[i]] = labels[label[i]]
        return res
    else:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        pred_sentences = [text]
        tf_batch = tokenizer(pred_sentences, max_length=128, padding=True, truncation=True, return_tensors='tf')
        model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
        model.load_weights('savedmodel')
        tf_outputs = model(tf_batch)
        tf_predictions = tf.nn.softmax(tf_outputs[0], axis=-1)
        labels = ['Negative', 'Positive']
        label = tf.argmax(tf_predictions, axis=1)
        label = label.numpy()
        res = {}
        for i in range(len(pred_sentences)):
            print(pred_sentences[i], ": \n", labels[label[i]])
            res["res"] = labels[label[i]]
        return res