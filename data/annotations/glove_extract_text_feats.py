from torchtext.vocab import GloVe
from torchtext.data.utils import get_tokenizer

glove = GloVe(name='6B')
tokenizer = get_tokenizer("basic_english")

word_list = ['hand', 'bowl', 'towel', 'some-nonexistent-word']
tokens = [w for w in tokenizer(' '.join(word_list)) if w in glove.stoi]

embeddings = glove.get_vecs_by_tokens(tokens)
