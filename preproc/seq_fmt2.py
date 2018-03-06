# -*- coding: utf-8 -*-
from __future__ import absolute_import
from .document import ParsedDocument,XmlDocument
from .corenlp import StanfordCoreNLP
from .annotation import KBPAnnMgr,ApfAnnMgr
import os
import codecs
import sys
from . import _list_files
def _sequence_tag_bio(doc):
    outlines = u''
    mentions= doc._annotate
    sentences = doc._text_spans
    for id,sentence in enumerate(sentences):
        tokens= sentence['tokens']
        tok_num = len(tokens)
        mention_tags = ['O']* tok_num
        coref_tags = ['-']*tok_num
        for mention in mentions:
            if mention['sent_id'] != id:
                continue
            mention_tokens= mention['mention_tokens']
            md_tag = mention['md_tag']
            coref_tag = mention['coref_tag']
            tokids=[]
            for token in mention_tokens:
                (sent, tok) = [int(d) for d in token.split('_')]
                if sent != id:
                    print 'mention cross sentence at {}'.format(sentence['origin_text'])
                    continue
                tokids.append(tok)
            for pos,tokid in enumerate(tokids):
                curr_md = md_tag
                curr_coref = coref_tag
                if pos ==0:
                    curr_md = 'B-' + curr_md
                else:
                    curr_md = 'I-' +curr_md
                if pos == 0:
                    curr_coref = '(' + curr_coref
                if pos == len(tokids) -1:
                    curr_coref = curr_coref + ')'
                if pos > 0 and pos < len(tokids) -1:
                    curr_coref = '-'
                if mention_tags[tokid] == 'O':
                    mention_tags[tokid] = curr_md
                    coref_tags[tokid]= curr_coref
        source =[]
        target =[]
        for token,mention,coref in zip(tokens,mention_tags, coref_tags):
            token_feature= [token['word_lower'].replace(u'#',u'@'), token['word'].replace(u'#',u'@'),
                            token['caps'].replace(u'#',u'@'), token['pos'].replace(u'#',u'@'),
                            token['ner'].replace(u'#',u'@')]
            if token.has_key(u'comb-word'):
                token_feature.append( token[u'comb-word'].replace(u'#',u'@'))
            source.append('#'.join(token_feature))
            target.append(mention)
        source = u' '.join(source)
        target = u' '.join(target)
        outlines += u'{}|||{} </s>\n'.format(source,target)
    return outlines

def build_tree_tag(mentions, tok_num):
    mentions.sort(cmp = lambda x,y:cmp(x[0], y[0]))
    tag_out=[('X',[],[]) for i in range(tok_num)]
    for mention in mentions:
        (start,end, mtype)= mention
        tag_out[start][1].append('('+mtype)
        tag_out[end][2].append(')'+mtype)
    otags=[]
    for tag in tag_out:
        pre= ' '.join(tag[1]).strip()
        suc =' '.join(tag[2][::-1]).strip()
        if pre != '':
            otags.append(pre)
        otags.append(tag[0])
        if suc != '':
            otags.append(suc)
    otags= ' '.join(otags)
    max_tag_num = max([len(x[1]) for x in tag_out])
    if max_tag_num >1:
        print 'nested tag:{}'.format(otags)
    
    return otags
        
def _sequence_tag_x(doc):
    outlines = u''
    mentions= doc._annotate
    sentences = doc._text_spans
    for id,sentence in enumerate(sentences):
        tokens= sentence['tokens']
        tok_num = len(tokens)
        curr_mentions = []
        for mention in mentions:
            if mention['sent_id'] != id:
                continue
            mention_tokens= mention['mention_tokens']
            md_tag = mention['md_tag']
            
            tok_start= int(mention_tokens[0].split('_')[1])
            tok_end = int(mention_tokens[-1].split('_')[1])
            curr_mentions.append((tok_start,tok_end, md_tag))
        
        target =build_tree_tag(curr_mentions, tok_num)
        source =[]
        for token in tokens:
            token_feature= [token['word_lower'].replace(u'#',u'@'), token['word'].replace(u'#',u'@'),
                            token['caps'].replace(u'#',u'@'), token['pos'].replace(u'#',u'@'),
                            token['ner'].replace(u'#',u'@')]
            if token.has_key(u'comb-word'):
                token_feature.append( token[u'comb-word'].replace(u'#',u'@'))
            source.append('#'.join(token_feature))
        source = u' '.join(source)
        
        outlines += u'{}|||{} </s>\n'.format(source,target.decode('utf-8'))
    return outlines    

#in format 'BIO' will ignore all nested tags,in format 'XX' will build tree sequence
def gen_sequence_tags(json_dir, outfile, fmt='BIO', encoding = 'utf-8'):
    fout= codecs.open(outfile, 'w', encoding= encoding)
    seqtag_func= None
    if fmt == 'BIO':
        seqtag_func= _sequence_tag_bio
    elif fmt =='XX':
        seqtag_func= _sequence_tag_x
    else:
        print 'unknown format {}'.format(fmt)
        return
    files = _list_files(json_dir, '.json')
    for f in files:
        print 'processing {}'.format(f)
        doc = ParsedDocument()
        doc.load(f)
        outlines = seqtag_func(doc)
        fout.write(outlines)
        fout.flush()
    fout.close()
