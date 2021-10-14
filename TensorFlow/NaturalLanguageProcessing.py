'''Natural Language Processing
: Field in computing that deals with communication between natural (human) languages and computer languages
i.e. Spellcheck, auto-complete, voice assistance

1) Recurrent Neural Networks (RNN)
- Sentiment Analysis
- Character Generation
'''

'''
Bag of words
> Convert every word into integer. Then analyze sentence and keep the track of frequency of word in a sentence, save its frequencies.
> We may loose the actual meaning of sentence since same amount of word could make different sentences depends on the location in the sentence.
'''
vocab = {} #maps word to integer representing it
word_encoding = 1
def bag_of_words(text):
    global word_encoding

    words = text.lower().split(" ") #create a list of all of the words in the text
    bag = {}                        #Stores all of the encodings and their frequency

    for word in words:
        if word in vocab:
            encoding = vocab[word]  #get encoding from vocab
        else:
            vocab[word] = word_encoding
            encoding = word_encoding
            word_encoding += 1

        if encoding in bag:
            bag[encoding] += 1
        else:
            bag[encoding] = 1
    return bag

text = "this is a test to see if t his test will work is is teat a a"
bag = bag_of_words(text)
print(bag)
print(vocab)
