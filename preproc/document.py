# -*- coding: utf-8 -*-
from __future__ import absolute_import
import sys
import os
from .corenlp import StanfordCoreNLP
import json
import re
import codecs
class Document(object):
    def __init__(self, fname, name = None):
        if name == None:
            name = fname
        self._name = name
        f = codecs.open(fname,encoding='utf-8')
        text = [line for line in f]
        text = u''.join(text)
        
        self._origin_text= text
        f.close()
        self._text_spans = []
        self._tag_spans= []
        self._extract_text()
    def _ignore(self, span, ignore_start, ignore_end):
        (char_begin, char_end, origin_text) = [span[u'char_begin'], span[u'char_end'], span[u'origin_text']]
        #no overlap
        if ignore_end <= char_begin or ignore_start >= char_end:
            return [span]
        elif char_begin >= ignore_start and char_end <= ignore_end:
            return []
        elif char_begin < ignore_start and char_end <= ignore_end:
            new_span = dict(char_begin = char_begin, char_end = ignore_start, origin_text = origin_text[:(ignore_start - char_begin)])
            return [new_span]
        elif char_begin >= ignore_start and char_end > ignore_end:
            new_span = dict(char_begin = ignore_end, char_end = char_end, origin_text = origin_text[(ignore_end-char_begin):])
            return [new_span]
        elif char_begin <  ignore_start and char_end > ignore_end:
            new_span1= dict(char_begin = char_begin, char_end = ignore_start, origin_text = origin_text[:(ignore_start - char_begin)])
            new_span2 = dict(char_begin = ignore_end, char_end = char_end, origin_text = origin_text[(ignore_end-char_begin):])
            return [new_span1, new_span2]
            
    def ignore_spans(self, spans_ignore):
        if spans_ignore is None:
            return
        for (ignore_start, ignore_end) in spans_ignore:
            text_spans = []
            for span in self._text_spans:
                text_spans += self._ignore(span, ignore_start, ignore_end)
            self._text_spans = text_spans
            tag_spans = []
            for span in self._tag_spans:
                tag_spans += self._ignore(span, ignore_start, ignore_end)
            self._tag_spans = tag_spans
                
                
    def _extract_text(self):
        char_begin = 0
        char_end = len(self._origin_text)
        self._text_spans.append( dict(char_begin = char_begin, char_end = char_end, origin_text = self._origin_text))
    
    @property
    def text_spans(self):
        return self._text_spans
    @property
    def tag_spans(self):
        return self._tag_spans

class XmlDocument(Document):
    PATTERN_DOCID = re.compile(r'<(doc|DOC) id="(.*)"')
    
    def __init__(self, ignore_tag_len = True, ignore_first_line = False, **kwargs):
        self._ignore_first_line = ignore_first_line
        self._ignore_tag_len = ignore_tag_len
        super(XmlDocument, self).__init__(**kwargs)
    def _extract_text(self):
        self._text_spans = []
        raw_text = self._origin_text
        if self._ignore_first_line:
            matchrlt= self.PATTERN_DOCID.search(raw_text)
            start= matchrlt.start()
            raw_text= raw_text[start:]
            self._origin_text= raw_text
        match = self.PATTERN_DOCID.search(raw_text)
        if match:
            self._name= match.groups()[1]
            
        
        curr_pos = 0
        curr_char_begin = 0
        end_pos = len(raw_text)
        while curr_pos < end_pos:
            if raw_text[curr_pos] == '<':
                next_pos = raw_text.find('>',  curr_pos+1) 
                if next_pos <0:
                    raise ValueError('xml tag ">" not matched')
                next_pos +=1
                if not self._ignore_tag_len:
                    length = next_pos - curr_pos
                    curr_span = dict(char_begin = curr_char_begin , 
                                     char_end = curr_char_begin +length,
                                     origin_text = raw_text[curr_pos:next_pos],
                                     in_tag = True)
                    curr_char_begin += length
                    self._tag_spans.append(curr_span)
                curr_pos = next_pos
            else:
                next_pos = raw_text.find('<', curr_pos)
                if next_pos <0:
                    next_pos = end_pos
                length = next_pos-curr_pos
                if len > 0:
                    curr_span = dict(char_begin = curr_char_begin , 
                                         char_end = curr_char_begin +length,
                                         origin_text = raw_text[curr_pos:next_pos],
                                         in_tag = False)
                    curr_char_begin += length
                    self._text_spans.append(curr_span)
                curr_pos = next_pos
def _proc_token(token):
    word = token['word']
    token['word_lower']= word.lower()
    caps= u'UNKNOWN'
    if all(c>chr(127) for c in word):
        caps=u'CHINESE'
    elif not word.isalpha():
        caps=u"NUM_PUNC"
    elif word.islower():
        caps= u'LOWER'
    elif word.isupper():
        caps=u'UPPER'
    elif word[0].isupper():
        caps=u'UPPER_FIRST'
    else:
        caps = u'MISC'
    token['caps'] = caps
    
    
class ParsedDocument(Document):
    def __init__(self,  document = None):
        self._annotate=[]
        self._tag_annotate=[]
        self._name = 'unknown'
        self._text_spans = []
        self._tag_spans = []
        self._origin_text= None
        if document:
            self._origin_text= document._origin_text
            self._name = document._name
            self._text_spans =document.text_spans
            self._tag_spans = document.tag_spans
            

    def load(self, file):
        fin = open(file)
        data = json.load(fin)
        self._name = data[u'name']
        self._text_spans = data[u'text_spans']
        self._tag_spans = data[u'tag_spans']
        self._annotate = data[u'annotate']
        self._tag_annotate = data[u'tag_annotate']
        self._origin_text = data[u'origin_text']
        
    def dump(self, file):
        fout = open(file, 'w')
        data = dict(name= self._name, text_spans = self._text_spans,
                    tag_spans= self._tag_spans, annotate= self._annotate,
                    tag_annotate= self._tag_annotate,
                    origin_text= self._origin_text)
        
        json.dump(data, fout, ensure_ascii=True, encoding='utf-8')
        fout.close()
        
    def is_overlap(self, s1,e1, s2,e2):
        return max(s1,s2) < min(e1,e2)
    #do coref with golden mention detect
    def set_md_tag(self):
        mentions= self._annotate
        sentences = self._text_spans
        for id,sentence in enumerate(sentences):
            tokens= sentence['tokens']
            for token in tokens:
                token['mention'] = 'O'
            
            for mention in mentions:
                if mention['sent_id'] != id:
                    continue
                mention_tokens= mention['mention_tokens']
                md_tag = mention['md_tag']
                
                tokids=[]
                for token in mention_tokens:
                    (sent, tok) = [int(d) for d in token.split('_')]
                    if sent != id:
                        print 'mention cross sentence at {}'.format(sentence['origin_text'])
                        continue
                    tokids.append(tok)
                for pos,tokid in enumerate(tokids):
                    curr_md = md_tag
                    
                    if pos ==0:
                        curr_md = 'B-' + curr_md
                    else:
                        curr_md = 'I-' +curr_md
                    if tokens[tokid]['mention'] =='O':
                        tokens[tokid]['mention']= curr_md

    def attach_annotation(self, annotation):
        out_anns = [] # annotation not in text spans
        self._annotate=[]
        self._tag_annotate=[]
        for mention in annotation.mentions:
            mention_tokens = []
           
            curr_sent_id = -1
            for sent_id, sentence in enumerate(self.text_spans):
                if not self.is_overlap(mention._char_begin, mention._char_end, 
                                       sentence[u'char_begin'], sentence[u'char_end']):
                    continue
                tokens = sentence['tokens']
                for tok_id, token in enumerate(tokens):
                    if self.is_overlap(mention._char_begin, mention._char_end, 
                                       token[u'char_begin'], token[u'char_end']):
                        mention_tokens.append('{}_{}'.format(sent_id, tok_id))
                        curr_sent_id = sent_id
                if len(mention_tokens) >0:
                    break
            
            if len(mention_tokens) == 0:
                out_anns.append(mention)
                continue
            md_tag = '{}_{}'.format(mention._entity_type, mention._mention_type)
            coref_tag = mention._coref_id
            mention_dict = dict(mention_tokens= mention_tokens, md_tag= md_tag,coref_tag= coref_tag, sent_id = curr_sent_id)
            self._annotate.append(mention_dict)
        for mention in out_anns:
            sent = -1
            for sent_id, sentence in enumerate(self._tag_spans):
                if self.is_overlap(mention._char_begin, mention._char_end, 
                                       sentence[u'char_begin'], sentence[u'char_end']): 
                    sent = sent_id
                    break
            md_tag = '{}_{}'.format(mention._entity_type, mention._mention_type)
            coref_tag = mention._coref_id
            mention_dict = dict(tag_id= sent, md_tag= md_tag,coref_tag= coref_tag,
                                char_begin=mention._char_begin,
                                char_end=mention._char_end)
            self._tag_annotate.append(mention_dict)
    
    '''
        this may cause sentences unmatch to annotator, so need done before annotation
    '''
    def split_sentence(self, max_sent_len = 200):
        new_spans = []
        for span in self._text_spans:           
            tokens = span['tokens']
            if len(tokens) < max_sent_len:
                new_spans.append(span)
                continue
            nsplit = len(tokens)/ max_sent_len +1
            num_per_sent = len(tokens)/nsplit +1
            for i in range(nsplit):
                start= i*num_per_sent
                end = start + num_per_sent
                if end > len(tokens):
                    end= len(tokens)
                curr_tokens= tokens[start:end]
                char_begin = curr_tokens[0][u'char_begin']
                char_end =  curr_tokens[-1][u'char_end']
                origin_text = span['origin_text']
                if self._origin_text is not None:
                    origin_text= self._origin_text[char_begin:char_end]
                sent_span = dict(char_begin= char_begin, char_end = char_end,
                                   origin_text = origin_text, tokens = curr_tokens)
                new_spans.append(sent_span)
        self._text_spans= new_spans
                
        pass
                    
    def parse_document(self,corenlp):
        sent_spans = []
        for span in self._text_spans:
            span_begin = span[u'char_begin']
            span_end = span[u'char_end']
            raw_text = span[u'origin_text']
            if raw_text.strip()=='':
                continue
            annotation = corenlp.annotate(raw_text)
            if annotation is None:
                continue
            for sentence in annotation:
                sent_begin = sentence[0][u'char_begin'] + span_begin
                sent_end = sentence[-1][u'char_end'] + span_begin
                
                sent_text =''
                for token in sentence:
                    token[u'char_begin'] += span_begin
                    token[u'char_end'] += span_begin
                    _proc_token(token)
                    sent_text +=token[u'origin_text'] + u' '
                if self._origin_text is not None:
                    sent_text =self._origin_text[sent_begin:sent_end]
                sent_span = dict(char_begin= sent_begin, char_end = sent_end,
                                   origin_text = sent_text, tokens = sentence)
                sent_spans.append(sent_span)
        self._text_spans = sent_spans


class TextDocument(Document):
    def __init__(self, text=None):
        self._annotate = []
        self._tag_annotate = []
        self._name = 'unknown'
        self._text_spans = []
        self._tag_spans = []
        self._origin_text = None
        if text:
            self._origin_text = text
            self._name = 'text'
            self._text_spans = [{'char_begin': 0, 'char_end': len(text), 'origin_text': text}]
            self._tag_spans = []

    def load(self, file):
        fin = open(file)
        data = json.load(fin)
        self._name = data[u'name']
        self._text_spans = data[u'text_spans']
        self._tag_spans = data[u'tag_spans']
        self._annotate = data[u'annotate']
        self._tag_annotate = data[u'tag_annotate']
        self._origin_text = data[u'origin_text']

    def dump(self, file):
        fout = open(file, 'w')
        data = dict(name=self._name, text_spans=self._text_spans,
                    tag_spans=self._tag_spans, annotate=self._annotate,
                    tag_annotate=self._tag_annotate,
                    origin_text=self._origin_text)

        json.dump(data, fout, ensure_ascii=True, encoding='utf-8')
        fout.close()

    def is_overlap(self, s1, e1, s2, e2):
        return max(s1, s2) < min(e1, e2)

    # do coref with golden mention detect
    def set_md_tag(self):
        mentions = self._annotate
        sentences = self._text_spans
        for id, sentence in enumerate(sentences):
            tokens = sentence['tokens']
            for token in tokens:
                token['mention'] = 'O'

            for mention in mentions:
                if mention['sent_id'] != id:
                    continue
                mention_tokens = mention['mention_tokens']
                md_tag = mention['md_tag']

                tokids = []
                for token in mention_tokens:
                    (sent, tok) = [int(d) for d in token.split('_')]
                    if sent != id:
                        print 'mention cross sentence at {}'.format(sentence['origin_text'])
                        continue
                    tokids.append(tok)
                for pos, tokid in enumerate(tokids):
                    curr_md = md_tag

                    if pos == 0:
                        curr_md = 'B-' + curr_md
                    else:
                        curr_md = 'I-' + curr_md
                    if tokens[tokid]['mention'] == 'O':
                        tokens[tokid]['mention'] = curr_md

    def attach_annotation(self, annotation):
        out_anns = []  # annotation not in text spans
        self._annotate = []
        self._tag_annotate = []
        for mention in annotation.mentions:
            mention_tokens = []

            curr_sent_id = -1
            for sent_id, sentence in enumerate(self.text_spans):
                if not self.is_overlap(mention._char_begin, mention._char_end,
                                       sentence[u'char_begin'], sentence[u'char_end']):
                    continue
                tokens = sentence['tokens']
                for tok_id, token in enumerate(tokens):
                    if self.is_overlap(mention._char_begin, mention._char_end,
                                       token[u'char_begin'], token[u'char_end']):
                        mention_tokens.append('{}_{}'.format(sent_id, tok_id))
                        curr_sent_id = sent_id
                        print 'match', token['origin_text'], token[u'char_begin'], token[u'char_end']
                        print mention._char_begin, mention._char_end, mention._mention
                if len(mention_tokens) > 0:
                    break

            if len(mention_tokens) == 0:
                out_anns.append(mention)
                continue
            md_tag = '{}_{}'.format(mention._entity_type, mention._mention_type)
            coref_tag = mention._coref_id
            mention_dict = dict(mention_tokens=mention_tokens, md_tag=md_tag, coref_tag=coref_tag, sent_id=curr_sent_id)
            self._annotate.append(mention_dict)
        for mention in out_anns:
            sent = -1
            for sent_id, sentence in enumerate(self._tag_spans):
                if self.is_overlap(mention._char_begin, mention._char_end,
                                   sentence[u'char_begin'], sentence[u'char_end']):
                    sent = sent_id
                    break
            md_tag = '{}_{}'.format(mention._entity_type, mention._mention_type)
            coref_tag = mention._coref_id
            mention_dict = dict(tag_id=sent, md_tag=md_tag, coref_tag=coref_tag,
                                char_begin=mention._char_begin,
                                char_end=mention._char_end)
            self._tag_annotate.append(mention_dict)

    '''
        this may cause sentences unmatch to annotator, so need done before annotation
    '''

    def split_sentence(self, max_sent_len=200):
        new_spans = []
        for span in self._text_spans:
            tokens = span['tokens']
            if len(tokens) < max_sent_len:
                new_spans.append(span)
                continue
            nsplit = len(tokens) / max_sent_len + 1
            num_per_sent = len(tokens) / nsplit + 1
            for i in range(nsplit):
                start = i * num_per_sent
                end = start + num_per_sent
                if end > len(tokens):
                    end = len(tokens)
                curr_tokens = tokens[start:end]
                char_begin = curr_tokens[0][u'char_begin']
                char_end = curr_tokens[-1][u'char_end']
                origin_text = span['origin_text']
                if self._origin_text is not None:
                    origin_text = self._origin_text[char_begin:char_end]
                sent_span = dict(char_begin=char_begin, char_end=char_end,
                                 origin_text=origin_text, tokens=curr_tokens)
                new_spans.append(sent_span)
        self._text_spans = new_spans

        pass

    def parse_document(self, corenlp):
        sent_spans = []
        for span in self._text_spans:
            span_begin = span[u'char_begin']
            span_end = span[u'char_end']
            raw_text = span[u'origin_text']
            if raw_text.strip() == '':
                continue
            # print 'raw-----', raw_text
            blank_num = len(raw_text) - len(raw_text.lstrip())
            annotation = corenlp.annotate(raw_text)
            if annotation is None:
                continue
            for sentence in annotation:
                sent_begin = sentence[0][u'char_begin'] + span_begin + blank_num
                sent_end = sentence[-1][u'char_end'] + span_begin + blank_num
                # print 'sent_begin:', sentence[0][u'char_begin'], 'sent_end:', sent_end
                sent_text = ''
                for token in sentence:
                    # if token['word'] == 'For':
                    #     print 'bingo', token['char_begin'], '-'+raw_text+'-'
                    # if token['word'] == 'Netanyahu':
                    #     print 'token:', token[u'char_begin'], token[u'char_end'], '-' + self._origin_text[sent_begin:sent_end] + '-', span_begin

                    token[u'char_begin'] += span_begin + blank_num
                    token[u'char_end'] += span_begin + blank_num
                    _proc_token(token)
                    sent_text += token[u'origin_text'] + u' '
                if self._origin_text is not None:
                    sent_text = self._origin_text[sent_begin:sent_end]
                sent_span = dict(char_begin=sent_begin, char_end=sent_end,
                                 origin_text=sent_text, tokens=sentence)
                sent_spans.append(sent_span)
        self._text_spans = sent_spans


            
            
        
    
                
                    
                
                
                    
        
        
        
        
        
        
        
