import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow import keras
from tensorflow.keras import layers

tfd = tfp.distributions

class AlphaBeta(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def call(self, inputs):
        mu, upsilon = inputs
        alpha = mu * upsilon2  # KW discretized outputs fail if a or b < 1
        beta = (1-mu) * upsilon2
        return alpha, beta
    
    
class BetaSampling(layers.Layer):
    def __init__(self, num_samples, **kwargs):
        super().__init__(**kwargs)
        self.num_samples = num_samples
        
    def call(self, inputs):
        alpha, beta = inputs
        # tf.print("alpha: ", alpha[:10], "beta: ", beta[:10])
        dist = tfd.Beta(alpha[:,0], beta[:,0])
        samples = tf.transpose(dist.sample(self.num_samples))*5  # (None, num_samples)
        samples = 0.5 + samples
        return samples
    
class BetaPoint(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def call(self, inputs):
        alpha, beta = inputs
        output = (alpha / (alpha + beta))[:,0]*5
        return tf.clip_by_value(output + 0.5, 1., 5.)  # (None,)
    
    
class KumaraswamySingleOutput(layers.Layer):  # Same as KumaraswamyMedian
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        alpha, beta = inputs
        pred = (1-2**(-1/beta))**(1/alpha)*5  # median
        # pred = ((alpha-1)/(alpha*beta-1))**(1/alpha)*5  # mode for a, b > 1
        # tf.print(pred[:2,:])
        return pred
    
class KumaraswamyDiscretizedMode(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        output = tf.argmax(inputs[:,:,1], axis=-1) + 1  # Might not work properly with bin_size!=1.
        return output
    
class KumaraswamyMedian(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        alpha, beta = inputs
        output = ((1 - 2**(-1/beta))**(1/alpha))*5
        output = tf.clip_by_value(output + 0.5, 1., 5.)
        return output

class KumaraswamyMode(layers.Layer):  # Mode only functions properly if alpha and beta exceed 1
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        alpha, beta = inputs
        output = ((alpha - 1) / (alpha*beta - 1))**(1/alpha)
        output = tf.math.real(output)*5.
        output = tf.clip_by_value(output + 0.5, 1., 5.)
        return output
    
class KumaraswamyDiscretizedOutputs(layers.Layer):
    def __init__(self, bin_size=1., min_rating=1, max_rating=5, **kwargs):
        # num_bins should be something divisible by 5 s.t. the model can achieve 0 loss
        super().__init__(**kwargs)
        self.bin_size = bin_size
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.num_bins = (self.max_rating-self.min_rating)/self.bin_size + 1
        self.bins = tf.range(self.min_rating, self.max_rating + bin_size, bin_size)
        self.bins_01 = tf.range(1, self.num_bins+1)/self.num_bins

    def cdf(self, x, a, b):
        return 1 - (1-x**a)**b
        
    def call(self, inputs):
        alpha, beta = inputs
        
        # Calculate the prediction for each bin
        preds = tf.ones_like(alpha) * self.bins
        # tf.print(self.bins, self.bins_01)
                
        # Calculate cdf at each bin end: (None, num_bins)
        cdf = self.cdf(self.bins_01[:-1], alpha, beta)
        # Replace last cdf bin to avoid issues with calculating d/dx of cdf with convex tails
        cdf = tf.concat([cdf, tf.ones_like(alpha)], axis=-1)
        # tf.print(cdf, cdf2)
        
        # Calculate mass in each bin: (None, num_bins)
        mass = tf.concat([cdf[:,:1], tf.experimental.numpy.diff(cdf)], axis=-1)

        # Output tensors: prediction, mass
        output = tf.concat([tf.expand_dims(preds, -1), tf.expand_dims(mass, -1)], axis=-1)
        # tf.print("preds", preds[:2], "ab", alpha[:2], beta[:2], "cdf", cdf[:2], "mass", mass[:2])
        return output
  
    
class RecEmbeddings(layers.Layer):
    def __init__(self, num_users, num_items, embedding_size, **kwargs):
        super().__init__()
        self.uid_features = layers.Embedding(num_users, embedding_size, name="uid_features", **kwargs)
        self.iid_features = layers.Embedding(num_items, embedding_size, name="iid_features", **kwargs)

    def call(self, inputs):
        uid_input, iid_input = inputs
        return self.uid_features(uid_input), self.iid_features(iid_input)
