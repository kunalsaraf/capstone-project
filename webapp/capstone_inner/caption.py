import numpy as np
from keras.layers import LSTM, Embedding, Dense, Dropout
from keras.layers.merge import add
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras import Input
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.sequence import pad_sequences

def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

def load_clean_descriptions(filename, dataset):
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
      # split line by white space
      tokens = line.split()
      # split id from description
      image_id, image_desc = tokens[0], tokens[1:]
      # skip images not in the set
      if image_id in dataset:
        # create list
        if image_id not in descriptions:
          descriptions[image_id] = list()
        # wrap description in tokens
        desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
        # store
        descriptions[image_id].append(desc)
    return descriptions

def load_set(filename):
	doc = load_doc(filename)
	dataset = list()
	# process line by line
	for line in doc.split('\n'):
		# skip empty lines
		if len(line) < 1:
			continue
		# get the image identifier
		identifier = line.split('.')[0]
		dataset.append(identifier)
	return set(dataset)

def get_vocab():
  filename ='./capstone_inner/data/Flickr_8k.trainImages.txt'
  train = load_set(filename)
  print('Dataset: %d' % len(train))
  # descriptions
  train_descriptions = load_clean_descriptions('./capstone_inner/data/descriptions.txt', train)
  print('Descriptions: train=%d' % len(train_descriptions))
  all_train_captions = []
  for key, val in train_descriptions.items():
      for cap in val:
          all_train_captions.append(cap)
  len(all_train_captions)
  word_count_threshold = 10
  word_counts = {}
  nsents = 0
  for sent in all_train_captions:
      nsents += 1
      for w in sent.split(' '):
          word_counts[w] = word_counts.get(w, 0) + 1
  vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
  return vocab

def get_word():
  vocab=get_vocab()
  ixtoword = {}
  wordtoix = {}
  ix = 1
  for w in vocab:
      wordtoix[w] = ix
      ixtoword[ix] = w
      ix += 1
  return wordtoix,ixtoword

def getModel(wordtoix,ixtoword):
  max_length=34
  vocab_size=1652
  embedding_dim=200
  embeddings_index = {} # empty dictionary
  f = open('./capstone_inner/model/glove.6B.200d.txt', encoding="utf-8")
  for line in f:
      values = line.split()
      word = values[0]
      coefs = np.asarray(values[1:], dtype='float32')
      embeddings_index[word] = coefs
  f.close()
  embedding_matrix = np.zeros((vocab_size, embedding_dim))
  for word, i in wordtoix.items():
      #if i < max_words:
      embedding_vector = embeddings_index.get(word)
      if embedding_vector is not None:
          # Words not found in the embedding index will be all zeros
          embedding_matrix[i] = embedding_vector
  print('Found %s word vectors.' % len(embeddings_index))
  inputs1 = Input(shape=(2048,))
  fe1 = Dropout(0.5)(inputs1)
  fe2 = Dense(256, activation='relu')(fe1)
  inputs2 = Input(shape=(max_length,))
  se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
  se2 = Dropout(0.5)(se1)
  se3 = LSTM(256)(se2)
  decoder1 = add([fe2, se3])
  decoder2 = Dense(256, activation='relu')(decoder1)
  outputs = Dense(vocab_size, activation='softmax')(decoder2)
  model = Model(inputs=[inputs1, inputs2], outputs=outputs)
  model.layers[2].set_weights([embedding_matrix])
  model.layers[2].trainable = False
  model.compile(loss='categorical_crossentropy', optimizer='adam')
  model.load_weights("./capstone_inner/model/model_final.h5")
  return model

def preprocess(image_path):
    # Convert all the images to size 299x299 as expected by the inception v3 model
    img = image.load_img(image_path, target_size=(299, 299))
    # Convert PIL image to numpy array of 3-dimensions
    x = image.img_to_array(img)
    # Add one more dimension
    x = np.expand_dims(x, axis=0)
    # preprocess the images using preprocess_input() from inception module
    x = preprocess_input(x)
    return x

def encode(image):
    model = InceptionV3(weights='imagenet')
    model_new = Model(model.input, model.layers[-2].output)
    image = preprocess(image) # preprocess the image
    fea_vec = model_new.predict(image) # Get the encoding vector for the image
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1]) # reshape from (1, 2048) to (2048, )
    return fea_vec

def greedySearch(photo,model,wordtoix,ixtoword):
    max_length=34
    in_text = 'startseq'

    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

def beam_search_pred(model, pic_fe, wordtoix,ixtoword,K_beams = 5, log = False):
    start_token = 'startseq'
    end_token = 'endseq'

    start = [wordtoix[start_token]]
    start_word = [[start, 0.0]]
    max_length=34
    
    while len(start_word[0][0]) < max_length:
        temp = []
        for s in start_word:
            sequence  = pad_sequences([s[0]], maxlen=max_length).reshape((1,max_length)) #sequence of most probable words 
                                                                                         # based on the previous steps
            preds = model.predict([pic_fe.reshape(1,2048), sequence])
            word_preds = np.argsort(preds[0])[-K_beams:] # sort predictions based on the probability, then take the last
                                                         # K_beams items. words with the most probs
            # Getting the top <K_beams>(n) predictions and creating a
            # new list so as to put them via the model again
            for w in word_preds:
                
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                if log:
                    prob += np.log(preds[0][w]) # assign a probability to each K words4
                else:
                    prob += preds[0][w]
                temp.append([next_cap, prob])
        start_word = temp
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])

        # Getting the top words
        start_word = start_word[-K_beams:]
    
    start_word = start_word[-1][0]
    captions_ = [ixtoword[i] for i in start_word]
    final_caption = []
    for i in captions_:
        if i != end_token:
            final_caption.append(i)
        else:
            break
    final_caption = ' '.join(final_caption[1:])
    return final_caption

def predict( modelImageCaptioning,test_image_path,wordtoix,ixtoword):
    img_new = encode(test_image_path).reshape(1, 2048)
    return greedySearch(img_new, modelImageCaptioning,wordtoix,ixtoword),beam_search_pred(modelImageCaptioning, img_new,wordtoix,ixtoword)

if __name__=="__main__":
    wordtoix, ixtoword = get_word()
    modelImageCaptioning = getModel(wordtoix,ixtoword)
    test_image_path = './capstone_inner/test/beach.jpg'
    greedyResult,beamResult=predict(modelImageCaptioning,test_image_path,wordtoix,ixtoword)
    print("Result by greedy = ",greedyResult)
    print("Result By Beam Search = ",beamResult)
