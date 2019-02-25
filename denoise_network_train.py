from common import *
from denoise_network import get_denoise_model
from get_data import get_data

EPOCHS = 50

seqs_train, seqs_test = get_data()

# get traning data
if os.path.exists('data/denoise_data/denoise_generator.npy') and os.path.exists('data/denoise_data/denoise_generator_val.npy'):
    denoise_generator     = np.load('data/denoise_data/denoise_generator.npy'    )
    denoise_generator_val = np.load('data/denoise_data/denoise_generator_val.npy')

else:
    denoise_generator     = DenoiseHPatches(seqs_train, batch_size=500)
    denoise_generator_val = DenoiseHPatches(seqs_test, batch_size=500)
    np.save('data/denoise_data/denoise_generator.npy'    , denoise_generator    )
    np.save('data/denoise_data/denoise_generator_val.npy', denoise_generator_val)


# get model
shape = (32, 32, 1)
denoise_model = get_denoise_model(shape)

# callbacks
callbacks = [
    keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5, verbose = 3, mode= 'auto'),
    keras.callbacks.ModelCheckpoint('data/denoise_model.weights.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=True)
]

# optimiser
opt  = opt = keras.optimizers.nadam()

# loss
loss = 'mean_absolute_error'

# train network
denoise_model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
denoise_history = denoise_model.fit_generator(generator=denoise_generator, epochs=EPOCHS, callbacks=callbacks,
                                              verbose=1, validation_data=denoise_generator_val)

# plot training curves

plt.plot(denoise_history.history['loss'])
plt.plot(denoise_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
