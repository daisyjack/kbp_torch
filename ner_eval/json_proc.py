import os
import sys
from preproc.document import ParsedDocument
import json
import re
import numpy
import urllib
import logging
# from .dec_encdec import EmsembleSearch,build_emsemble_searcher
#
# from NER.json_data import NERData
logger = logging.getLogger(__name__)
def mask_src(source):
    src_len= [len(a) for a in source]
    src_dim = source[0].shape[1]
    source_mat = numpy.zeros((len(source), max(src_len), src_dim), dtype='int32')
    source_mask = numpy.zeros((len(source), max(src_len)), dtype='float32')
    for i, src in enumerate(source):
        source_mat[i, :src_len[i],:] = src
        source_mask[i, :src_len[i]] = 1
    return (source_mat, source_mask)
# class EncDecJsonEvaluator(object):
#     def __init__(self, config, models):
#         self._searcher = build_emsemble_searcher(config, models,
#                                                  config['BeamSize'])
#         self._out_vocab = config['OutTags']
#         self._data_gen = NERData(config)
#         self._eol_symbol = config['eol_symbol']
#         self._batchsize= config['BatchSize']
#
#     def process(self, document):
#         document._annotate=[]
#         src_list=[]
#         total_recs = []
#         for id, sentence in enumerate(document.text_spans):
#             tokens= sentence['tokens']
#             feaMat= self._data_gen.get_feature(tokens)
#             src_list.append(feaMat)
#             if len(src_list) >= self._batchsize:
#                 (source,source_mask) = mask_src(src_list)
#                 recs, costs= self._searcher.batch_decode(source, source_mask,
#                                                 self._eol_symbol)
#                 total_recs += recs
#                 src_list=[]
#         if len(src_list) > 0:
#             (source,source_mask) = mask_src(src_list)
#             recs, costs= self._searcher.batch_decode(source, source_mask,
#                                             self._eol_symbol)
#             total_recs += recs
#         for id, rec in enumerate(total_recs):
#             rec = [self._out_vocab.getWord(r) for r in rec]
#             annotate= self._extract_mentions(rec, id)
#             document._annotate += annotate
#
#     def _extract_mentions(self, tokens, sent_id):
#         mentions=[]
#         pos = 0
#         mention_stack =[]
#         unmatch = 0
#         for token in tokens:
#             if token =='X':
#                 pos +=1
#             elif token.startswith('('):
#                 mention_stack.append((token[1:], pos))
#             elif token.startswith(')'):
#                 mention_type = token[1:]
#                 is_match = False
#                 for pre in mention_stack[::-1]:
#                     if pre[0] == mention_type:
#                         is_match=True
#                         mention_stack.remove(pre)
#                         mtoks=[]
#                         for tokid in range(pre[1], pos):
#                             mtoks.append('{}_{}'.format(sent_id, tokid))
#                         mention = dict(mention_tokens=mtoks, md_tag= mention_type,
#                                    coref_tag= 'unknown',sent_id=sent_id)
#                         mentions.append(mention)
#                         break
#                 if not is_match:
#                     unmatch+=1
#         unmatch+= len(mention_stack)
#         if unmatch >0:
#             logger.warn('unmatched sequence {}'.format(' '.join(tokens)))
#         return mentions
    

    
        
