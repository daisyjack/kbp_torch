# -*- coding: utf-8 -*-
from __future__ import absolute_import
import sys
import json, requests
import codecs
import urllib
from .langconv import *
class StanfordCoreNLP(object):
    def __init__(self, lang = 'eng', server_url = 'http://127.0.0.1:9000'):
        if server_url[-1] == '/':
            server_url = server_url[:-1]
        self.server_url = server_url
        self._lang = lang
    
    def annotate(self, text):
        if self._lang == 'eng':
            return self._annotate_en(text)
        elif self._lang =='spa':
            return self._annotate_sp(text)
        elif self._lang =='cmn':
            return self._annotate_ch(text)
        else:
            raise NotImplemented('unknown language {}'. format(self._lang))
    def _split_char(self, token):
        word= token['word']
        if all(c<chr(127) for c in word):
            token[u'comb-word'] = token['word']
            token[u'origin_text']= token['word']
            return [token]
        tokens = []
        start= token['char_begin']
        for pos,char in enumerate(word):
            ntoken = token.copy()
            ntoken[u'char_begin']= start +pos
            ntoken[u'char_end']= start + pos +1
            ntoken[u'comb-word'] = token['word']
            ntoken[u'word'] = char
            ntoken[u'origin_text']=ntoken[u'word']
            tokens.append(ntoken)
        return tokens
    def split_chinese_char(self, sentences):
        new_sents=[]
        for sent in sentences:
            tokens = sent
            new_tokens = []
            for token in tokens:                
                new_tokens+= self._split_char(token)
            new_sents.append(new_tokens)
        return new_sents    
        
    def _annotate_en(self, text):
        text= text.replace(u'’', u'\'')
        text= text.replace(u'-', u' ')
        text= text.replace(u'/', u' ')
        properties = {}
        properties['annotators'] = 'tokenize, ssplit, pos, lemma, ner'
        return self._annotate(text, properties)
    def _annotate_sp(self, text):
        properties = {}
        properties['annotators'] = 'tokenize, ssplit, pos,  ner'
        properties['tokenize.language'] = 'es'
        properties['pos.model'] = 'edu/stanford/nlp/models/pos-tagger/spanish/spanish-distsim.tagger'
        properties['ner.model'] = 'edu/stanford/nlp/models/ner/spanish.ancora.distsim.s512.crf.ser.gz'
        properties['ner.applyNumericClassifiers'] = 'false'
        properties['ner.useSUTime'] = 'false'
        #text= text.replace(u'-', u' ')
        #text= text.replace(u'/', u' ')
        sents= self._annotate(text, properties)
        if sents is None:
            return None
        return self.modifi_spanish(sents)
    
    def _annotate_ch(self,text):
        text_jian = Converter('zh-hans').convert(text)
        if len(text_jian) != len(text):
            print 'traditional to simple not equal {}:{}'.format(len(text),len(text_jian))
            print text_jian.encode('utf-8')
            print text.encode('utf-8')
            sys.exit(0)
        text= text_jian
        properties = {}
        properties['annotators'] = 'tokenize, ssplit, pos, ner'
        properties['customAnnotatorClass.tokenize'] = 'edu.stanford.nlp.pipeline.ChineseSegmenterAnnotator'
        properties['tokenize.model'] = 'edu/stanford/nlp/models/segmenter/chinese/ctb.gz'
        properties['tokenize.sighanCorporaDict'] = 'edu/stanford/nlp/models/segmenter/chinese'
        properties['tokenize.serDictionary'] = 'edu/stanford/nlp/models/segmenter/chinese/dict-chris6.ser.gz'
        properties['tokenize.sighanPostProcessing'] = 'true'

        #properties['ssplit.boundaryTokenRegex'] = urllib.quote_plus('[.]|[!?]+|[。]|[！？]+')
        properties['ssplit.boundaryTokenRegex'] = urllib.quote_plus('[!?]+|[。]|[！？]+')
        properties['pos.model'] = 'edu/stanford/nlp/models/pos-tagger/chinese-distsim/chinese-distsim.tagger'
        properties['ner.model'] = 'edu/stanford/nlp/models/ner/chinese.misc.distsim.crf.ser.gz'
        properties['ner.applyNumericClassifiers'] = 'false'
        properties['ner.useSUTime'] = 'false'
        text= text.replace(u'-', u' ')
        text= text.replace(u'/', u' ')
        sents = self._annotate(text, properties)
        if sents is None:
            return None
        #return sents
        return self.split_chinese_char(sents)
        

    def _annotate(self, text, properties=None):
        if not properties:
            properties = {}
        properties['outputFormat'] = 'json'
        properties['inputFormat']='text'
        # Checks that the Stanford CoreNLP server is started.
        try:
            requests.get(self.server_url).ok == True
        except requests.exceptions.ConnectionError:
            raise Exception('Check whether you have started the CoreNLP server e.g.\n'
            '$ cd stanford-corenlp-full-2015-12-09/ \n'
            '$ java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer')
        #headers={'content-type':'application/x-www-form-urlencoded;charset=UTF-8'}
        #headers={'content-type':'application/x-www-form-urlencoded'}
        #print text.encode('utf-8')
        '''f=open('debug1.txt','w')
        f.write(text.encode('utf-8'))
        f.close()'''
        r = requests.post(
            self.server_url, params={
                'properties': str(properties)
            }, data=text.encode('utf-8'))#,headers=headers)
        r.encoding='UTF-8'
        output = r.text
        return self.get_annotation(output)
    def modifi_spanish(self, sentences):
        for sent in sentences:
            tokens = sent
            for pos in range(len(tokens)-1):
                if tokens[pos]['char_begin']== tokens[pos+1]['char_begin']:
                    rtokens=[]
                    (start,end)= (tokens[pos]['char_begin'],tokens[pos]['char_end'])
                    for p in range(pos, len(tokens)):
                        if (tokens[p]['char_begin'], tokens[p]['char_end'])== (start,end):
                            rtokens.append(tokens[p])
                        else:
                            break
                    for p in range(len(rtokens)-1):
                        rtokens[p]['char_end'] = rtokens[p]['char_begin'] + len(rtokens[p]['origin_text'])
                        rtokens[p+1]['char_begin']= rtokens[p]['char_end']
                        
                        
        return sentences 
        
    @staticmethod
    def get_annotation(json_str):
        sentences=[]
        try:
            sentences = json.loads(json_str.encode('utf-8'))[u'sentences']
        except ValueError,ex:
            #print 'illegal result \"{}\"'.format(json_str.encode('utf-8'))
            print ex
            return None
            
        sents_obj = []
        for sentence in sentences:
            sent_ =[]
            tokens= sentence[u'tokens']
            for token in tokens:
                word = token[u'word'].replace(u' ', u'_')
                char_begin = token[u'characterOffsetBegin']
                char_end = token[u'characterOffsetEnd']
                pos = token[u'pos']
                ner = token[u'ner']
                origin_text = token[u'originalText']
                sent_.append(dict(word=word, pos = pos, ner= ner, char_begin = char_begin, char_end = char_end, origin_text = origin_text))
            sents_obj.append(sent_)
        return sents_obj    
