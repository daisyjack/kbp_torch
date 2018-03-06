# -*- coding: utf-8 -*-
from __future__ import absolute_import
import sys
import codecs
import re
import os
import sgmllib
from xml.dom.minidom import parse
import xml.dom.minidom
class Mention(object):
    def __init__(self, char_begin, char_end, entity_type, mention_type, coref, mention=None):
        self._char_begin = char_begin
        self._char_end = char_end
        self._entity_type = entity_type
        self._mention_type= mention_type
        self._coref_id = coref
        self._mention = mention

class DocAnnotation(object):
    def __init__(self, name):
        self._name = name
        self._mentions = []
    @property
    def mentions(self):
        return self._mentions
    def sort_mentions(self):
        self._mentions.sort(cmp = lambda x,y: cmp(x._char_begin, y._char_begin))
    def add_mention(self, mention = None, **kwargs):
        if mention is not None:
            self._mentions.append(mention)
        else:
            self._mentions.append(Mention(**kwargs))

class KBPAnnMgr(object):
    Pat_Split = re.compile(r':|\-')
    def __init__(self, tabfile, col_title=3, col_link=4, col_ner= 5, col_mt= 6):
        self._col_title= col_title
        self._col_link= col_link
        self._col_ner= col_ner
        self._col_mt = col_mt
        self._doc_anns = dict()
        self._load_tabfile(tabfile)
    def get_annotation(self, name):
        if not self._doc_anns.has_key(name):
            return None
        return self._doc_anns[name]
    @property
    def doc_anns(self):
        return self._doc_anns
    def _parse_title(self, title):
        parts = self.Pat_Split.split(title)
        if len(parts)< 3:
            return (None,None, None)
        return (parts[0], int(parts[1]),int(parts[2]))
    def _load_tabfile(self, tabfile):
        fin = open(tabfile)
        for line in fin:
            parts= line.split('\t')
            (name, char_begin, char_end) = self._parse_title(parts[self._col_title])
            if name is None:
                print 'illegal line "{}"'.format(line)
                continue
            char_end +=1
            ner = parts[self._col_ner]
            mt = parts[self._col_mt]
            mention_text= parts[2].decode('utf-8')
            coref = parts[self._col_link]
            if not self._doc_anns.has_key(name):
                self._doc_anns[name] = DocAnnotation(name)
            self._doc_anns[name].add_mention(Mention(char_begin, char_end, 
                                                    ner, mt, coref,mention_text))
        fin.close()
        for name in self._doc_anns.keys():
            self._doc_anns[name].sort_mentions()

class ApfAnnMgr(object):
    def __init__(self, apflist):
        self._doc_anns = dict()
        with open(apflist) as flist:
            for line in flist:
                line= line.strip()
                annotation = self._load_apf(line)
                self._doc_anns[annotation._name] = annotation
                
                
    def get_annotation(self, name):
            if not self._doc_anns.has_key(name):
                return None
            return self._doc_anns[name]        
    
    def _load_apf(self, apffile):
        domtree= xml.dom.minidom.parse(apffile)
        src_tag = domtree.documentElement.getElementsByTagName('source_file')[0]
        name = src_tag.getAttribute('URI')
        annotation = DocAnnotation(name)
        entities = domtree.documentElement.getElementsByTagName('entity')
        entity_id = 0
        set_entities = {'PER', 'ORG', 'GPE', 'FAC', 'LOC'}
        set_entclass= {'SPC'}
        set_mentions={'NAM', 'NOM', 'PRE'}
        for entity in entities:
            ent_type = entity.getAttribute('TYPE')
            sub_type = entity.getAttribute('SUBTYPE')
            ent_class = entity.getAttribute('CLASS')
            mentions= entity.getElementsByTagName('entity_mention')
            if ent_type not in set_entities or ent_class not in set_entclass:
                continue
            for mention in mentions:
                mention_type = mention.getAttribute('TYPE')
                if mention_type not in set_mentions:
                    continue
                head = mention.getElementsByTagName('head')[0]
                charseq = head.getElementsByTagName('charseq')[0]
                char_begin=int(charseq.getAttribute('START'))
                char_end = int(charseq.getAttribute('END'))
                char_end +=1
                annotation.add_mention(Mention(char_begin, char_end, 
                                              ent_type, 
                                              mention_type, 
                                              str(entity_id)))
            entity_id +=1
        return annotation
        
   
        
    
            
            
            
        
        
        
