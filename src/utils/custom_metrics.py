import tensorflow as tf
from tensorflow.keras import backend as K # type: ignore

# Custom Precision metric
class Precision(tf.keras.metrics.Metric):
    def __init__(self, name='precision', **kwargs):
        super(Precision, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.predicted_positives = self.add_weight(name='pp', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=1)
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)

        true_positives = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred), self.dtype))
        predicted_positives = tf.reduce_sum(tf.cast(tf.not_equal(y_pred, 0), self.dtype))

        self.true_positives.assign_add(true_positives)
        self.predicted_positives.assign_add(predicted_positives)
    
    def result(self):
        return self.true_positives / (self.predicted_positives + K.epsilon())
    
    def reset_states(self):
        self.true_positives.assign(0)
        self.predicted_positives.assign(0)

# Custom Recall metric
class Recall(tf.keras.metrics.Metric):
    def __init__(self, name='recall', **kwargs):
        super(Recall, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.possible_positives = self.add_weight(name='pp', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=1)
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)

        true_positives = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred), self.dtype))
        possible_positives = tf.reduce_sum(tf.cast(tf.not_equal(y_true, 0), self.dtype))

        self.true_positives.assign_add(true_positives)
        self.possible_positives.assign_add(possible_positives)
    
    def result(self):
        return self.true_positives / (self.possible_positives + K.epsilon())
    
    def reset_states(self):
        self.true_positives.assign(0)
        self.possible_positives.assign(0)

# Custom F1 Score metric
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = Precision()
        self.recall = Recall()
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred)
        self.recall.update_state(y_true, y_pred)
    
    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
    
    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()
