import common

# denoise model
def get_denoise_model(shape):

  inputs = Input(shape)
  conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
                 #kernel_regularizer=regularizers.l2(0.0001),activity_regularizer=regularizers.l1(0.00001),

  pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
  ## Bottleneck
  conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
                 #kernel_regularizer=regularizers.l2(0.0001), activity_regularizer=regularizers.l1(0.00001),


  ## Now the decoder starts
  up3 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv2))
               #kernel_regularizer=regularizers.l2(0.0001), activity_regularizer=regularizers.l1(0.00001),

  merge3 = concatenate([conv1,up3], axis = -1)
  conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge3)
                 #kernel_regularizer=regularizers.l2(0.0001), activity_regularizer=regularizers.l1(0.00001),


  conv4 = Conv2D(1, 3,  padding = 'same')(conv3)

  shallow_unet = Model(inputs = inputs, outputs = conv4)
  return shallow_unet

def get_denoise_model_add(shape):

  inputs = Input(shape)
  conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
                 #kernel_regularizer=regularizers.l2(0.0001),activity_regularizer=regularizers.l1(0.00001),

  pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
  ## Bottleneck
  conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
                 #kernel_regularizer=regularizers.l2(0.0001), activity_regularizer=regularizers.l1(0.00001),


  ## Now the decoder starts
  up3 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv2))
               #kernel_regularizer=regularizers.l2(0.0001), activity_regularizer=regularizers.l1(0.00001),

  merge3 = Add([conv1,up3], axis = -1)
  conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge3)
                 #kernel_regularizer=regularizers.l2(0.0001), activity_regularizer=regularizers.l1(0.00001),


  conv4 = Conv2D(1, 3,  padding = 'same')(conv3)

  shallow_unet = Model(inputs = inputs, outputs = conv4)
  return shallow_unet
