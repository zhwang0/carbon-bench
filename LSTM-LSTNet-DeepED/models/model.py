import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from configs.constant_glob import *

class MM_LSTM_Age_v51(keras.Model):
     # same as MM_LSTM_Age_v5, but move to 
    def __init__(self, n_out, n_age_fea=15, n_cons=None, n_lstm_eco=1, n_lstm_height=1):
        super(MM_LSTM_Age_v51, self).__init__()
        self.n_out = n_out
        self.n_age_fea = n_age_fea
        self.n_cons = n_cons
        self.n_lstm_eco = n_lstm_eco
        self.n_lstm_height = n_lstm_height
        self.ls_maximum = [-1.0516589, -0.64791672, -0.99871467, -0.86651843, -0.72868288, -0.72231681, -0.78230592]

        # Ecosystem LSTM model
        self.m_ecos = tf.keras.Sequential()
        for _ in range(self.n_lstm_eco):
            self.m_ecos.add(layers.LSTM(256, return_sequences=True))
            
        self.ls_branch_lstms = []
        for i in range(n_out):
            self.ls_branch_lstms.append([
                layers.LSTM(256, return_sequences=True, name='data'+str(i)),
                layers.LSTM(256, return_sequences=True, activation='sigmoid', name='att'+str(i)),
                layers.LSTM(256, return_sequences=True, name='tsOut'+str(i))
            ])

        # Dense layers for processing the LSTM outputs
        self.ls_dense1 = [layers.Dense(256, activation='relu', name='dense1'+str(i)) for i in range(n_out)]
        self.ls_dense2 = [layers.Dense(64, activation='relu', name='dense2'+str(i)) for i in range(n_out)]
        self.ls_dense_final = [layers.Dense(1, name='output'+str(i)) for i in range(n_out)]

    def call(self, inputs):
        # 11 is compatiable with longer sequence
        x_input_init = inputs[:,11,self.n_cons:(self.n_cons+self.n_out)] 
        x_input_age = inputs[:,11,(self.n_cons+self.n_out):(self.n_cons+self.n_out+self.n_age_fea)]
        x_input_ecos = inputs[:,:,:self.n_cons]

        # Process through ecosystem LSTM model
        x_ecos = self.m_ecos(x_input_ecos)

        ls_outputs = []
        for i in range(self.n_out):
            sub_layers = self.ls_branch_lstms[i]
            
            cur_init = x_input_init[:,i:(i+1)]
            cur_age = layers.concatenate([
                x_input_age[:,:1],
                x_input_age[:,(i+1):(i+2)],
                x_input_age[:,(i+8):(i+9)]
            ])
            cur_init = tf.repeat(cur_init[:,tf.newaxis], tf.shape(x_ecos)[1], axis=1)
            cur_age = tf.repeat(cur_age[:,tf.newaxis], tf.shape(x_ecos)[1], axis=1)
            

            cur_out = sub_layers[0](x_ecos)
            cur_att = sub_layers[1](x_ecos)
            cur_out = cur_out * cur_att
            cur_out = sub_layers[2](cur_out)
            
            cur_out = self.ls_dense1[i](cur_out)
            cur_out = self.ls_dense2[i](cur_out)
            cur_out = layers.concatenate([cur_out, cur_age])
            cur_out = self.ls_dense_final[i](cur_out)
            cur_out = cur_out + cur_init
            cur_out = tf.maximum(cur_out, self.ls_maximum[i])
            ls_outputs.append(cur_out)

        output_all = layers.concatenate(ls_outputs) # [BATCH_SIZE*Year, year_len*24, n_out]
        output_all = output_all[:,11::12] # only output end of year [BATCH_SIZE*Year, year_len, n_out]
        return output_all
    
    
class MM_LSTM_Age_v5(keras.Model):
     # same as mm_lstm_age_test: add all age triplets into the second last layer
    def __init__(self, n_out, n_age_fea=15, n_cons=None, n_lstm_eco=1, n_lstm_height=1):
        super(MM_LSTM_Age_v5, self).__init__()
        self.n_out = n_out
        self.n_age_fea = n_age_fea
        self.n_cons = n_cons
        self.n_lstm_eco = n_lstm_eco
        self.n_lstm_height = n_lstm_height
        self.ls_maximum = [-1.0516589, -0.64791672, -0.99871467, -0.86651843, -0.72868288, -0.72231681, -0.78230592]

        # Ecosystem LSTM model
        self.m_ecos = tf.keras.Sequential()
        for _ in range(self.n_lstm_eco):
            self.m_ecos.add(layers.LSTM(256, return_sequences=True))
            
        self.ls_branch_lstms = []
        for i in range(n_out):
            self.ls_branch_lstms.append([
                layers.LSTM(256, return_sequences=True, name='data'+str(i)),
                layers.LSTM(256, return_sequences=True, activation='sigmoid', name='att'+str(i)),
                layers.LSTM(256, return_sequences=False, name='tsOut'+str(i))
            ])

        # Dense layers for processing the LSTM outputs
        self.ls_dense1 = [layers.Dense(256, activation='relu', name='dense1'+str(i)) for i in range(n_out)]
        self.ls_dense2 = [layers.Dense(64, activation='relu', name='dense2'+str(i)) for i in range(n_out)]
        self.ls_dense_final = [layers.Dense(1, name='output'+str(i)) for i in range(n_out)]

    def call(self, inputs):
        # 11 is compatiable with longer sequence
        x_input_init = inputs[:,11,self.n_cons:(self.n_cons+self.n_out)] 
        x_input_age = inputs[:,11,(self.n_cons+self.n_out):(self.n_cons+self.n_out+self.n_age_fea)]
        x_input_ecos = inputs[:,:,:self.n_cons]

        # Process through ecosystem LSTM model
        x_ecos = self.m_ecos(x_input_ecos)

        ls_outputs = []
        for i in range(self.n_out):
            sub_layers = self.ls_branch_lstms[i]
            cur_init = x_input_init[:,i:(i+1)]
            cur_age = layers.concatenate([
                x_input_age[:,:1],
                x_input_age[:,(i+1):(i+2)],
                x_input_age[:,(i+8):(i+9)]
            ])

            cur_out = sub_layers[0](x_ecos)
            cur_att = sub_layers[1](x_ecos)
            cur_out = cur_out * cur_att
            cur_out = sub_layers[2](cur_out)
            
            cur_out = self.ls_dense1[i](cur_out)
            cur_out = self.ls_dense2[i](cur_out)
            cur_out = layers.concatenate([cur_out, cur_age])
            cur_out = self.ls_dense_final[i](cur_out)
            cur_out = cur_out + cur_init
            cur_out = tf.maximum(cur_out, self.ls_maximum[i])
            ls_outputs.append(cur_out)

        output_all = layers.concatenate(ls_outputs)
        return output_all
    

class MM_LSTM_Age_v4(keras.Model):
     # same as mm_lstm_age_test: add all age triplets into the second last layer
    def __init__(self, n_out, n_age_fea=15, n_cons=None, n_lstm_eco=1, n_lstm_height=1):
        super(MM_LSTM_Age_v4, self).__init__()
        self.n_out = n_out
        self.n_age_fea = n_age_fea
        self.n_cons = n_cons
        self.n_lstm_eco = n_lstm_eco
        self.n_lstm_height = n_lstm_height
        self.ls_maximum = [-1.0516589, -0.64791672, -0.99871467, -0.86651843, -0.72868288, -0.72231681, -0.78230592]

        # Ecosystem LSTM model
        self.m_ecos = tf.keras.Sequential()
        for _ in range(self.n_lstm_eco):
            self.m_ecos.add(layers.LSTM(256, return_sequences=True))
            
        self.ls_branch_lstms = []
        for i in range(n_out):
            self.ls_branch_lstms.append([
                layers.LSTM(256, return_sequences=True, name='data'+str(i)),
                layers.LSTM(256, return_sequences=True, activation='sigmoid', name='att'+str(i)),
                layers.LSTM(256, return_sequences=False, name='tsOut'+str(i))
            ])

        # Dense layers for processing the LSTM outputs
        self.ls_dense1 = [layers.Dense(256, activation='relu', name='dense1'+str(i)) for i in range(n_out)]
        self.ls_dense2 = [layers.Dense(64, activation='relu', name='dense2'+str(i)) for i in range(n_out)]
        self.ls_dense_final = [layers.Dense(1, name='output'+str(i)) for i in range(n_out)]

    def call(self, inputs):
        x_input_init = inputs[:, :, self.n_cons:(self.n_cons+self.n_out)]
        x_input_age = inputs[:, :, (self.n_cons+self.n_out):(self.n_cons+self.n_out+self.n_age_fea)]
        x_input_ecos = inputs[:, :, :self.n_cons]
        x_inputs = layers.concatenate([x_input_ecos, x_input_age])

        # Process through ecosystem LSTM model
        x_ecos = self.m_ecos(x_inputs)

        ls_outputs = []
        for i in range(self.n_out):
            sub_layers = self.ls_branch_lstms[i]
            cur_init = x_input_init[:, :, i:(i+1)]
            cur_age = layers.concatenate([
                x_input_age[:, :, :1],
                x_input_age[:, :, (i+1):(i+2)],
                x_input_age[:, :, (i+8):(i+9)]
            ])
            cur_input = layers.concatenate([cur_init, cur_age, x_ecos])

            cur_out = sub_layers[0](cur_input)
            cur_att = sub_layers[1](cur_input)
            cur_out = cur_out * cur_att
            cur_out = sub_layers[2](cur_out)
            
            cur_out = self.ls_dense1[i](cur_out)
            cur_out = self.ls_dense2[i](cur_out)
            cur_out = layers.concatenate([cur_out, cur_input[:, -1, 1:4]])
            cur_out = self.ls_dense_final[i](cur_out)
            cur_out = cur_out + cur_input[:, -1, :1]
            cur_out = tf.maximum(cur_out, self.ls_maximum[i])
            ls_outputs.append(cur_out)

        output_all = layers.concatenate(ls_outputs)
        return output_all
  
class MM_LSTM_Age_v2(keras.Model):
     # same as mm_lstm_age_test: add all age triplets into the second last layer
    def __init__(self, n_out, n_age_fea=15, n_cons=None, n_lstm_eco=1, n_lstm_height=1):
        super(MM_LSTM_Age_v2, self).__init__()
        self.n_out = n_out
        self.n_age_fea = n_age_fea
        self.n_cons = n_cons
        self.n_lstm_eco = n_lstm_eco
        self.n_lstm_height = n_lstm_height

        # Ecosystem LSTM model
        self.m_ecos = tf.keras.Sequential()
        for _ in range(self.n_lstm_eco):
            self.m_ecos.add(layers.LSTM(256, return_sequences=True))
            
        self.ls_branch_lstms = []
        for i in range(n_out):
            self.ls_branch_lstms.append([
                layers.LSTM(256, return_sequences=True, name='data'+str(i)),
                layers.LSTM(256, return_sequences=True, activation='sigmoid', name='att'+str(i)),
                layers.LSTM(256, return_sequences=False, name='tsOut'+str(i))
            ])

        # Dense layers for processing the LSTM outputs
        self.ls_dense1 = [layers.Dense(256, activation='relu', name='dense1'+str(i)) for i in range(n_out)]
        self.ls_dense2 = [layers.Dense(64, activation='relu', name='dense2'+str(i)) for i in range(n_out)]
        self.ls_dense_final = [layers.Dense(1, name='output'+str(i)) for i in range(n_out)]

    def call(self, inputs):
        x_input_init = inputs[:, :, self.n_cons:(self.n_cons+self.n_out)]
        x_input_age = inputs[:, :, (self.n_cons+self.n_out):(self.n_cons+self.n_out+self.n_age_fea)]
        x_input_ecos = inputs[:, :, :self.n_cons]
        x_inputs = layers.concatenate([x_input_ecos, x_input_age])

        # Process through ecosystem LSTM model
        x_ecos = self.m_ecos(x_inputs)

        ls_outputs = []
        for i in range(self.n_out):
            sub_layers = self.ls_branch_lstms[i]
            cur_init = x_input_init[:, :, i:(i+1)]
            cur_age = layers.concatenate([
                x_input_age[:, :, :1],
                x_input_age[:, :, (i+1):(i+2)],
                x_input_age[:, :, (i+8):(i+9)]
            ])
            cur_input = layers.concatenate([cur_init, cur_age, x_ecos])

            cur_out = sub_layers[0](cur_input)
            cur_att = sub_layers[1](cur_input)
            cur_out = cur_out * cur_att
            cur_out = sub_layers[2](cur_out)
            
            cur_out = self.ls_dense1[i](cur_out)
            cur_out = self.ls_dense2[i](cur_out)
            cur_out = layers.concatenate([cur_out, cur_input[:, -1, 1:4]])
            cur_out = self.ls_dense_final[i](cur_out)
            cur_out = cur_out + cur_input[:, -1, :1]
            ls_outputs.append(cur_out)

        output_all = layers.concatenate(ls_outputs)
        return output_all

class DeepEDv2_LSTM(keras.Model):
     # same as mm_lstm_age_test: add all age triplets into the second last layer
    def __init__(self, n_out, n_age_fea=15, n_cons=None, n_lstm_eco=1, n_lstm_height=1):
        super(DeepEDv2_LSTM, self).__init__()
        self.n_out = n_out
        self.n_age_fea = n_age_fea
        self.n_cons = n_cons
        self.n_lstm_eco = n_lstm_eco
        self.n_lstm_height = n_lstm_height

        # Ecosystem LSTM model
        self.ecos_ts = layers.LSTM(256, return_sequences=False)
            
        # Dense layers for processing the LSTM outputs
        self.out = layers.Dense(7, name='output')

    def call(self, inputs):
        x_input_init = inputs[:, 11, self.n_cons:(self.n_cons+self.n_out)]
        x_input_ecos = inputs[:, :, :self.n_cons]

        x = self.ecos_ts(x_input_ecos)
        x = layers.concatenate([x_input_init, x])
        x = self.out(x)
        
        return x
    

class DeepEDv2_LSTMa1(tf.keras.Model):
  def __init__(self, n_out, n_age_fea=15, n_cons=None, n_lstm_eco=1, n_lstm_height=1):
    super(DeepEDv2_LSTMa1, self).__init__()
    self.n_out = n_out
    self.n_age_fea = n_age_fea
    self.n_cons = n_cons
    self.n_lstm_eco = n_lstm_eco
    self.n_lstm_height = n_lstm_height

    # Embedding layers
    self.encoder_embedding = layers.Dense(256, activation='relu')
    self.decoder_embedding = layers.Dense(256, activation='relu')

    # Encoder LSTM
    self.encoder_lstm_int = layers.LSTM(256, return_sequences=True)
    self.encoder_lstm_att = layers.LSTM(256, return_sequences=True, activation='sigmoid')
    self.encoder_lstm_out = layers.LSTM(256, return_sequences=False, return_state=True)

    # Decoder LSTM
    self.decoder_lstm = layers.LSTM(256, return_sequences=False)

    # Output dense layer
    self.out = layers.Dense(n_out)

  def call(self, inputs):
    x_input_init = inputs[:, 11:12, self.n_cons:(self.n_cons+self.n_out)]
    x_input_ecos = inputs[:, :, :self.n_cons]
    
    # Encoder
    encoder_embedded = self.encoder_embedding(x_input_ecos)
    encoder_x = self.encoder_lstm_int(encoder_embedded)
    encoder_w = self.encoder_lstm_att(encoder_embedded)
    encoder_x = encoder_x * encoder_w
    encoder_x, state_h, state_c = self.encoder_lstm_out(encoder_x)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_embedded = self.decoder_embedding(x_input_init)
    decoder_outputs = self.decoder_lstm(decoder_embedded, initial_state=encoder_states)

    # Output
    outputs = self.out(decoder_outputs)

    return outputs

class DeepEDv2_LSTMa(tf.keras.Model):
  def __init__(self, n_out, n_age_fea=15, n_cons=None, n_lstm_eco=1, n_lstm_height=1):
    super(DeepEDv2_LSTMa, self).__init__()
    self.n_out = n_out
    self.n_age_fea = n_age_fea
    self.n_cons = n_cons
    self.n_lstm_eco = n_lstm_eco
    self.n_lstm_height = n_lstm_height

    # Embedding layers
    self.encoder_embedding = layers.Dense(256, activation='relu')
    self.decoder_embedding = layers.Dense(256, activation='relu')

    # Encoder LSTM
    self.encoder_lstm = layers.LSTM(256, return_sequences=True, return_state=True)

    # Attention mechanism
    self.attention = layers.AdditiveAttention()

    # Decoder LSTM
    self.decoder_lstm = layers.LSTM(256, return_sequences=True, return_state=True)

    # Output dense layer
    self.out = layers.Dense(n_out)

  def call(self, inputs):
    x_input_init = inputs[:, 11:12, self.n_cons:(self.n_cons+self.n_out)]
    x_input_ecos = inputs[:, :, :self.n_cons]

    # Encoder
    encoder_embedded = self.encoder_embedding(x_input_ecos)
    encoder_outputs, state_h, state_c = self.encoder_lstm(encoder_embedded)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_embedded = self.decoder_embedding(x_input_init)
    decoder_outputs, _, _ = self.decoder_lstm(decoder_embedded, initial_state=encoder_states)

    # Attention
    context_vector, attention_weights = self.attention([decoder_outputs, encoder_outputs], return_attention_scores=True)
    decoder_combined_context = tf.concat([context_vector, decoder_outputs], axis=-1)

    # Output
    outputs = self.out(decoder_combined_context)

    return tf.squeeze(outputs)

    
class DeepEDv2_LSTNet_2d(keras.Model):
     # same as mm_lstm_age_test: add all age triplets into the second last layer
    def __init__(self, n_out, n_age_fea=15, n_cons=None, n_lstm_eco=1, n_lstm_height=1):
        super(DeepEDv2_LSTNet_2d, self).__init__()
        self.n_out = n_out
        self.n_age_fea = n_age_fea
        self.n_cons = n_cons
        self.n_lstm_eco = n_lstm_eco
        self.n_lstm_height = n_lstm_height

        # Ecosystem LSTM model
        # self.conv1d = layers.Conv1D(256, 6, padding='same', activation='relu')
        self.conv2d = layers.Conv2D(256, kernel_size=(6, self.n_cons), activation='relu')
        self.m_ecos = layers.LSTM(256, return_sequences=False)
            
        # Dense layers for processing the LSTM outputs
        self.out = layers.Dense(7, name='output')

    def call(self, inputs):
        x_input_init = inputs[:, 11, self.n_cons:(self.n_cons+self.n_out)]
        x_input_ecos = inputs[:, :, :self.n_cons]
        
        x_ecos = tf.reshape(x_input_ecos, (-1, x_input_ecos.shape[1], x_input_ecos.shape[2], 1))
        x_ecos = self.conv2d(x_ecos)
        x_ecos = tf.squeeze(x_ecos)
        x_ecos = self.m_ecos(x_ecos)
        x = layers.concatenate([x_input_init, x_ecos])
        x = self.out(x)
        
        return x
    

class DeepEDv2_LSTNet_1d(keras.Model):
     # same as mm_lstm_age_test: add all age triplets into the second last layer
    def __init__(self, n_out, n_age_fea=15, n_cons=None, n_lstm_eco=1, n_lstm_height=1):
        super(DeepEDv2_LSTNet_1d, self).__init__()
        self.n_out = n_out
        self.n_age_fea = n_age_fea
        self.n_cons = n_cons
        self.n_lstm_eco = n_lstm_eco
        self.n_lstm_height = n_lstm_height

        # Ecosystem LSTM model
        self.conv1 = layers.Conv1D(256, 6, padding='same', activation='relu')
        self.m_ecos = layers.LSTM(256, return_sequences=False)
            
        # Dense layers for processing the LSTM outputs
        self.out = layers.Dense(7, name='output')

    def call(self, inputs):
        x_input_init = inputs[:, 11, self.n_cons:(self.n_cons+self.n_out)]
        x_input_ecos = inputs[:, :, :self.n_cons]

        x_ecos = self.conv1(x_input_ecos)
        x_ecos = self.m_ecos(x_ecos)
        x = layers.concatenate([x_input_init, x_ecos])
        x = self.out(x)
        
        return x

class MM_LSTM_Age_v2_old(keras.Model):
     # same as mm_lstm_age_test: add all age triplets into the second last layer
    def __init__(self, n_out, n_age_fea=15, n_cons=None, n_lstm_eco=1, n_lstm_height=1):
        super(MM_LSTM_Age_v2_old, self).__init__()
        self.n_out = n_out
        self.n_age_fea = n_age_fea
        self.n_cons = n_cons
        self.n_lstm_eco = n_lstm_eco
        self.n_lstm_height = n_lstm_height

        # Ecosystem LSTM model
        self.m_ecos = tf.keras.Sequential()
        for _ in range(self.n_lstm_eco):
            self.m_ecos.add(layers.LSTM(256, return_sequences=True))

        self.ls_data_lstms = [layers.LSTM(256, return_sequences=True, name='data'+str(i)) for i in range(n_out)]
        self.ls_att_lstms = [layers.LSTM(256, return_sequences=True, activation='sigmoid', name='att'+str(i)) for i in range(n_out)]
        self.ls_ts_out_lstms = [layers.LSTM(256, return_sequences=False, name='tsOut'+str(i)) for i in range(n_out)]

        # Dense layers for processing the LSTM outputs
        self.ls_dense1 = [layers.Dense(256, activation='relu', name='dense1'+str(i)) for i in range(n_out)]
        self.ls_dense2 = [layers.Dense(64, activation='relu', name='dense2'+str(i)) for i in range(n_out)]
        self.ls_dense_final = [layers.Dense(1, name='output'+str(i)) for i in range(n_out)]

    def call(self, inputs):
        x_input_init = inputs[:, :, self.n_cons:(self.n_cons+self.n_out)]
        x_input_age = inputs[:, :, (self.n_cons+self.n_out):(self.n_cons+self.n_out+self.n_age_fea)]
        x_input_ecos = inputs[:, :, :self.n_cons]
        x_inputs = layers.concatenate([x_input_ecos, x_input_age])

        # Process through ecosystem LSTM model
        x_ecos = self.m_ecos(x_inputs)

        ls_outputs = []
        for i in range(self.n_out):
            cur_init = x_input_init[:, :, i:(i+1)]
            cur_age = layers.concatenate([
                x_input_age[:, :, :1],
                x_input_age[:, :, (i+1):(i+2)],
                x_input_age[:, :, (i+8):(i+9)]
            ])
            cur_input = layers.concatenate([cur_init, cur_age, x_ecos])

            cur_out = self.ls_data_lstms[i](cur_input)
            cur_att = self.ls_att_lstms[i](cur_input)
            cur_out = cur_out * cur_att

            # Final processing and dense layers
            cur_out = self.ls_ts_out_lstms[i](cur_out)
            cur_out = self.ls_dense1[i](cur_out)
            cur_out = self.ls_dense2[i](cur_out)
            cur_out = layers.concatenate([cur_out, cur_input[:, -1, 1:4]])
            cur_out = self.ls_dense_final[i](cur_out)
            cur_out = cur_out + cur_input[:, -1, :1]
            ls_outputs.append(cur_out)

        output_all = layers.concatenate(ls_outputs)
        return output_all
    
    
def init_state(batch_size, hidden_dim, num_outputs):
    # Initialize the states for the ecosystem model 
    h_state = [tf.zeros((batch_size, hidden_dim))]
    c_state = [tf.zeros((batch_size, hidden_dim))]
    
    # Initialize the states for each output-related set of LSTM layers
    for _ in range(num_outputs):
        h_state.append([tf.zeros((batch_size, hidden_dim))]*3)  
        c_state.append([tf.zeros((batch_size, hidden_dim))]*3)  

    return h_state, c_state

class MM_LSTM_Age_v3(keras.Model):
    def __init__(self, n_out, n_age_fea=15, n_cons=None):
        super(MM_LSTM_Age_v3, self).__init__()
        self.n_out = n_out
        self.n_age_fea = n_age_fea
        self.n_cons = n_cons
        
        self.m_ecos = layers.LSTM(256, return_sequences=True, return_state=True)

        self.ls_branch_lstms = []
        for i in range(n_out):
            sub_layers = [
                layers.LSTM(256, return_sequences=True, return_state=True, name='data'+str(i)),
                layers.LSTM(256, return_sequences=True, return_state=True, activation='sigmoid', name='att'+str(i)),
                layers.LSTM(256, return_sequences=False, return_state=True, name='tsOut'+str(i))
            ]
            self.ls_branch_lstms.append(sub_layers)
            
        self.ls_dense1 = [layers.Dense(256, activation='relu', name='dense1'+str(i)) for i in range(n_out)]
        self.ls_dense2 = [layers.Dense(64, activation='relu', name='dense2'+str(i)) for i in range(n_out)]
        self.ls_dense_final = [layers.Dense(1, name='output'+str(i)) for i in range(n_out)]


    def call(self, inputs, h_state=None, c_state=None):
        if h_state is None or c_state is None:
            batch_size = tf.shape(inputs)[0]
            h_state, c_state = init_state(batch_size, hidden_dim=256, num_outputs=self.n_out)
            
        x_input_init = inputs[:, :, self.n_cons:(self.n_cons+self.n_out)]
        x_input_age = inputs[:, :, (self.n_cons+self.n_out):(self.n_cons+self.n_out+15)]
        x_input_ecos = inputs[:, :, :self.n_cons]
        x_inputs = layers.concatenate([x_input_ecos, x_input_age])
        
        x_ecos, h_eco, c_eco = self.m_ecos(x_inputs, initial_state=[h_state.pop(0), c_state.pop(0)])
        ls_outputs = []
        ls_h_state = [h_eco]
        ls_c_state = [c_eco]
        
        for i in range(self.n_out):
            sub_layers = self.ls_branch_lstms[i]
            cur_init = x_input_init[:, :, i:(i+1)]
            cur_age = layers.concatenate([
                x_input_age[:, :, :1], 
                x_input_age[:, :, (i+1):(i+2)], 
                x_input_age[:, :, (i+8):(i+9)]
            ])
            cur_input = layers.concatenate([cur_init, cur_age, x_ecos])
            
            cur_out, h1, c1 = sub_layers[0](cur_input, initial_state=[h_state[i][0], c_state[i][0]])
            cur_att, h2, c2 = sub_layers[1](cur_input, initial_state=[h_state[i][1], c_state[i][1]])
            cur_out = cur_out * cur_att
            cur_out, h3, c3 = sub_layers[2](cur_out, initial_state=[h_state[i][2], c_state[i][2]])

            cur_out = self.ls_dense1[i](cur_out)
            cur_out = self.ls_dense2[i](cur_out)
            cur_out = layers.concatenate([cur_out, cur_input[:, -1, 1:4]])
            cur_out = self.ls_dense_final[i](cur_out)
            cur_out = cur_out + cur_input[:, -1, :1]
            
            ls_h_state.append([h1, h2, h3])
            ls_c_state.append([c1, c2, c3])
            ls_outputs.append(cur_out)
        
        output_all = layers.concatenate(ls_outputs)
        return output_all, ls_h_state, ls_c_state


class MM_LSTM_Age_v3_nonNeg(keras.Model):
    def __init__(self, n_out, n_age_fea=15, n_cons=None):
        super(MM_LSTM_Age_v3_nonNeg, self).__init__()
        self.n_out = n_out
        self.n_age_fea = n_age_fea
        self.n_cons = n_cons
        self.ls_maximum = [-1.0516589, -0.64791672, -0.99871467, -0.86651843, -0.72868288, -0.72231681, -0.78230592]
        
        self.m_ecos = layers.LSTM(256, return_sequences=True, return_state=True)

        self.ls_branch_lstms = []
        for i in range(n_out):
            sub_layers = [
                layers.LSTM(256, return_sequences=True, return_state=True, name='data'+str(i)),
                layers.LSTM(256, return_sequences=True, return_state=True, activation='sigmoid', name='att'+str(i)),
                layers.LSTM(256, return_sequences=False, return_state=True, name='tsOut'+str(i))
            ]
            self.ls_branch_lstms.append(sub_layers)
            
        self.ls_dense1 = [layers.Dense(256, activation='relu', name='dense1'+str(i)) for i in range(n_out)]
        self.ls_dense2 = [layers.Dense(64, activation='relu', name='dense2'+str(i)) for i in range(n_out)]
        self.ls_dense_final = [layers.Dense(1, name='output'+str(i)) for i in range(n_out)]


    def call(self, inputs, h_state=None, c_state=None):
        if h_state is None or c_state is None:
            batch_size = tf.shape(inputs)[0]
            h_state, c_state = init_state(batch_size, hidden_dim=256, num_outputs=self.n_out)
            
        x_input_init = inputs[:, :, self.n_cons:(self.n_cons+self.n_out)]
        x_input_age = inputs[:, :, (self.n_cons+self.n_out):(self.n_cons+self.n_out+15)]
        x_input_ecos = inputs[:, :, :self.n_cons]
        x_inputs = layers.concatenate([x_input_ecos, x_input_age])
        
        x_ecos, h_eco, c_eco = self.m_ecos(x_inputs, initial_state=[h_state.pop(0), c_state.pop(0)])
        ls_outputs = []
        ls_h_state = [h_eco]
        ls_c_state = [c_eco]
        
        for i in range(self.n_out):
            sub_layers = self.ls_branch_lstms[i]
            cur_init = x_input_init[:, :, i:(i+1)]
            cur_age = layers.concatenate([
                x_input_age[:, :, :1], 
                x_input_age[:, :, (i+1):(i+2)], 
                x_input_age[:, :, (i+8):(i+9)]
            ])
            cur_input = layers.concatenate([cur_init, cur_age, x_ecos])
            
            cur_out, h1, c1 = sub_layers[0](cur_input, initial_state=[h_state[i][0], c_state[i][0]])
            cur_att, h2, c2 = sub_layers[1](cur_input, initial_state=[h_state[i][1], c_state[i][1]])
            cur_out = cur_out * cur_att
            cur_out, h3, c3 = sub_layers[2](cur_out, initial_state=[h_state[i][2], c_state[i][2]])

            cur_out = self.ls_dense1[i](cur_out)
            cur_out = self.ls_dense2[i](cur_out)
            cur_out = layers.concatenate([cur_out, cur_input[:, -1, 1:4]])
            cur_out = self.ls_dense_final[i](cur_out)
            cur_out = cur_out + cur_input[:, -1, :1]
            cur_out = tf.maximum(cur_out, self.ls_maximum[i])
            
            ls_h_state.append([h1, h2, h3])
            ls_c_state.append([c1, c2, c3])
            ls_outputs.append(cur_out)
        
        output_all = layers.concatenate(ls_outputs)
        return output_all, ls_h_state, ls_c_state

    