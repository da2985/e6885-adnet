class ADNET_v3(tf.keras.Model):

    ACTION_DIM = 11
    K = 10
    CONF_SCORE_DIM = 2


    def __init__(self):
        super(ADNET_v3, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters = 96, kernel_size = (7, 7), strides = (2, 2), padding = 'VALID', activation = 'relu', name = 'conv_1')
        self.max1  = tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides = (1, 1), padding = 'VALID')
        self.conv2 = tf.keras.layers.Conv2D(filters = 256, kernel_size = (5, 5), strides = (2, 2), padding = 'VALID', activation = 'relu', name = 'conv_2')
        self.max2  = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides = (2, 2), padding = 'VALID')
        self.conv3 = tf.keras.layers.Conv2D(filters = 512, kernel_size = (3, 3), strides = (2, 2), padding = 'VALID', activation = 'relu', name = 'conv_3')
        self.max3  = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides = (1, 1), padding = 'VALID')
        
        self.fc1 = tf.keras.layers.Conv2D(filters = 512, kernel_size = (3, 3), padding = 'VALID', activation = 'relu', name = 'fc1')
        self.rnn = CustomRNN()
        
    def build(self,action_history):
      super(ADNET_v3, self).build((None, 112, 112, 3))
      self.rnn.setActionHistory(action_history)

    def call(self, input_tensor, training=False):
        x = self.conv1(input_tensor)
        x = self.max1(x)
        x = self.conv2(x)
        x = self.max2(x)
        x = self.conv3(x)
        x = self.max3(x)

        x = self.fc1(x)
        action,confidence = self.rnn(x)
        return [action, confidence]

    def callForAction(self, input_tensor, training=False):
      '''
      Wrapper action for model.call() to only output action probabilities.
      For RL purposes, this is the only relevant output (as of 12/2/2021).
      '''
      return self.call(input_tensor, training)[0]

    def clearActionHistory(self):
      self.rnn.clearActionHistory()

    def compile(self, optimizer):
      super().compile(optimizer, loss={
         'output_1':tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
         'output_2': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
         })
      
    def debugModelSummary(self):
      '''
      call model.debugModelSummary().summary() to get around the inconvenience 
      from model.summary() returning 'multiple' for each layer's output shape
      '''
      dummyInput = tf.keras.layers.Input(shape = (112,112,3))
      return tf.keras.Model(inputs=[dummyInput], outputs = self.call(dummyInput))

class CustomRNN(layers.Layer):
    def __init__(self):
        super(CustomRNN, self).__init__()
        self.projection_1 = tf.keras.layers.Conv2D(filters = 256, kernel_size = (1, 1), padding = 'VALID')
        self.projection_2 = tf.keras.layers.Conv2D(filters = 256, kernel_size = (1, 1), padding = 'VALID')
        self.action_history = tf.zeros(shape = (1,1,1,256))
        self.action_classifier = tf.keras.layers.Conv2D(filters = 11, kernel_size = (1,1), padding = 'VALID',activation="softmax")
        self.confidence_classifier = tf.keras.layers.Conv2D(filters = 2, kernel_size = (1,1), padding = 'VALID',trainable=False,activation="softmax")

    def call(self, input):
        h = self.projection_1(self.action_history)
        y = h + self.projection_2(input)
        new_action_history = tf.math.tanh(y)
        self.setActionHistory(new_action_history)
        action = self.action_classifier(y)
        confidence = self.confidence_classifier(y)
        return action,confidence

    def setActionHistory(self, action_history):
      self.action_history = action_history
    
    def clearActionHistory(self):
      self.setActionHistory(tf.zeros((1, 1, 1, 256)))