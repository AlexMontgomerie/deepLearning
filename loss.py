import common

# Triplet loss
def triplet_loss(x):
  output_dim = 128
  a, p, n = x
  _alpha = 1.0
  positive_distance = K.mean(K.square(a - p), axis=-1)
  negative_distance = K.mean(K.square(a - n), axis=-1)
  return K.expand_dims(K.maximum(0.0, positive_distance - negative_distance + _alpha), axis = 1)

# SSIM Loss
def ssim_loss(x):
  clean = x[0]
  noisy = x[1]

  return 1 - tf.image.ssim(clean, noisy, max_val=1.0)
