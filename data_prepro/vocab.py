import sys
class  Vocab(object):
    def __init__(self, vocfile, unk_id = 0, padding_id = 0):
        self._word2id = {}
        self._id2word={}
        self.unk = '<UNK>'
        self.PADDING='<PADDING>'
        self.unk_id = unk_id
        self.padding_id = padding_id
        self._voc_name = vocfile
        with open(vocfile) as f:
            for line in f:
                parts = line.split()
                if len(parts) != 2:
                    print 'illegal voc line %s' %line
                    continue
                id = int(parts[1])
                self._word2id[parts[0]] = id
                self._id2word[id] = parts[0]
        self._vocab_size = max(self._word2id.values()) +1
        self.unk = self._id2word[self.unk_id]
        self.PADDING= self._id2word[self.padding_id]
        if self._vocab_size != len(self._word2id):
            print 'in vocab file {}, vocab_max {} not equal to vocab count {}, maybe empty id or others'\
                  .format(vocfile, self._vocab_size, len(self._word2id))
    def __str__(self):
        return self._voc_name
    def getVocSize(self):
        return self._vocab_size
    def getWord(self, id):
        return self._id2word[id] if self._id2word.has_key(id) else self.unk
    def getID(self, word):
        return self._word2id[word] if self._word2id.has_key(word) else self.unk_id
    @property
    def word2id(self):
        return self._word2id
    @property
    def id2word(self):
        return self._id2word            
    
