def evaluate_model(trainX, trainy, testX, testy):
 verbose, epochs, batch_size = 0, 10, 32
 n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
 model = Sequential()
 model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
 model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
 model.add(Dropout(0.5))
 model.add(MaxPooling1D(pool_size=2))
 model.add(Flatten())
 model.add(Dense(100, activation='relu'))
 model.add(Dense(n_outputs, activation='softmax'))
 model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
 # fit network
 model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
 # evaluate model
 _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
 return accuracy

X_word = [ text_to_word_sequence(x) for x in X_text ]
tk = Tokenizer()
tk.fit_on_texts(X_word)
X_token = tk.texts_to_sequences(X_word)
X_token[:2]

vocab_size = len(tk.word_index)
vocab_size

X_token_pad = pad_sequences(X_token, dtype=float, padding='post', maxlen=maxlen)

embedding_size = 30
maxlen = 60

def build_model_nlp():
  model = Sequential([
     layers.Embedding(input_dim=vocab_size+1, output_dim=embedding_size, input_length=maxlen, mask_zero=True),
     layers.Conv1D(20, kernel_size=15,padding='same', activation='relu'),
     layers.Conv1D(20, kernel_size=10,padding='same', activation='relu'),
     layers.Flatten(),
     layers.Dense(10, activation='relu'),
     layers.Dense(1, activation='linear')


  ])
  return model


tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
vocab_size = len(tokenizer.word_index)+1

encoded_X_train = tokenizer.texts_to_sequences(X_train)
encoded_X_test = tokenizer.texts_to_sequences(X_test)

encoded_X_train = sequence.pad_sequences(encoded_X_train, maxlen=33)
encoded_X_test = sequence.pad_sequences(encoded_X_test, maxlen=33)

input1 = Input(shape=(33,))
x = Embedding(input_dim=vocab_size, output_dim=32, input_length=33)(input1)
x = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(x)
x = MaxPooling1D(pool_size=2)(x)
x = Dropout(0.5)(x)
x = Flatten()(x)
x = Dense(30, activation='sigmoid')(x)
x = Dropout(0.5)(x)
x = Dense(5, activation='sigmoid')(x)
x = Dropout(0.5)(x)
output1 = Dense(1, activation='sigmoid')(x)

model = Model(input1, output1)
