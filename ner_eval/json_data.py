import torch
import os
import sys
from preproc.document import ParsedDocument
import json
import re
import numpy
import urllib
# from NER.dec_emsemble import ModelCon, EmsembleSearcher
from bio_eval_ner import infer_eval, eva_one_sentence, batch_eva_one_sentence
from encoder import LoadEmbedding
from bio_model import BioCNNEncoder, BioRnnDecoder, CMNBioCNNEncoder
import codecs
from process_data import AnnotationBatchGetter
# from coref.coref_emsemble import ModelCoref, CorefEmsemble

def write_kbp_file(document, fout, start=0, teamid= 'USCT_NELSLIP', garbage='TEDL16_EVAL'):
    origin_text= document._origin_text
    docstart=start
    name = document._name
    for mention in document._annotate:
        #mention_dict = dict(mention_tokens= mention_tokens,
        #md_tag= md_tag,coref_tag= coref_tag, sent_id = curr_sent_id)
        sentid = mention['sent_id']
        token_start= int(mention['mention_tokens'][0].split('_')[1])
        token_end= int(mention['mention_tokens'][-1].split('_')[1])
        sentence= document._text_spans[sentid]
        char_begin = sentence['tokens'][token_start]['char_begin']
        char_end = sentence['tokens'][token_end]['char_end']-1
        name_pos = '{}:{}-{}'.format(name, char_begin, char_end)
        text= origin_text[char_begin:char_end+1].encode('utf-8')
        text = text.replace('\t', ' ').replace('\n', ' ')
        coref= 'M{}.{}'.format(docstart,mention['coref_tag'])
        mdtag= mention['md_tag'].split('_')
        ner= mdtag[0]
        mentiontype= mdtag[1]
        confidence = 1.0

        gartag= '%s_%06d'%(garbage, start)
        start +=1
        line= '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(teamid, gartag, text,
                                          name_pos,coref,ner, mentiontype,
                                          confidence)
        fout.write(line)
    for tagmention in document._tag_annotate:
        '''mention_dict = dict(tag_id= sent, md_tag= md_tag,coref_tag= coref_tag,
                                char_begin=mention._char_begin,
                                char_end=mention._char_end)'''
        char_begin = tagmention['char_begin']
        char_end= tagmention['char_end']-1
        name_pos = '{}:{}-{}'.format(name, char_begin, char_end)
        text= origin_text[char_begin:char_end+1].encode('utf-8')
        text = text.replace('\t', ' ').replace('\n', ' ')
        coref= tagmention['coref_tag']
        mdtag= tagmention['md_tag'].split('_')
        ner= mdtag[0]
        mentiontype= mdtag[1]
        confidence = 1.0
        gartag= '%s_%06d'%(garbage, start)
        start +=1
        line= '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(teamid, gartag, text,
                                          name_pos,coref,ner, mentiontype,
                                          confidence)
        fout.write(line)
    fout.flush()
    return start
                                          
                                          

        
        
        
        
        
    
    

def extract_tag_mentions(document):
    document._tag_annotate = []
    pattern_author= re.compile(u'author=\"(.*?)\"', re.I)
    mentions =[]
    for id,tag_span in enumerate(document._tag_spans):
        text= tag_span['origin_text']
        start = tag_span['char_begin']
        authors = pattern_author.findall(text)
        for author in authors:
            begin = text.index(author) + start
            end= begin + len(author)
            author = author.encode('utf-8').replace(' ','').replace('\n','').replace('\t','')
            author= urllib.quote_plus(author)
            coref = 'Author.{}'.format(author)
            mention =  dict(tag_id= id, md_tag= 'PER_NAM',coref_tag= coref,
                                char_begin=begin,
                                char_end=end)
            mentions.append(mention)
    document._tag_annotate= mentions
                                       
            
        
        
        
# class CorefEvaluator(object):
#     def __init__(self, config, models):
#         self._models = []
#         self._config = config
#         for model in models:
#             self._models.append(ModelCoref(config, model))
#         self._emsemble = CorefEmsemble(self._models)
#         self._data_gen = CorefData(config)
#     def do_process(self, doc_parts, anaphore_mask):
#         return self._emsemble.process(doc_parts, anaphore_mask)
#     def process(self, document):
#         document.set_md_tag()
#         sentences = document._text_spans
#         mentions = document._annotate
#         if len(mentions) == 0:
#             return
#         (fea_parts,anaphore_mask) =self._data_gen.gen_feature(sentences,mentions)
#         (corefs, confidence) =  self.do_process(fea_parts, anaphore_mask)
#         for id,mention in enumerate(mentions):
#             coref_id = corefs[id]
#             mention['coref_tag'] = coref_id
        
        
    
class NEREvaluator(object):
    def __init__(self, config, models):
        self._models = []
        self._config = config
        # for model in models:
        #     self._models.append(ModelCon(config, model))
        # self._searcher = EmsembleSearcher(self._models, config['BeamSize'])
        self._out_vocab = config['BioOutTags']
        self._data_gen = NERData(config)
        self._batch_data_gen = AnnotationBatchGetter(config, 10)
        self.build_model(config)

    def build_model(self, config):
        if config['lang'] == 'cmn':
            word_emb = LoadEmbedding(config['eval_word_emb'])
            char_emb = LoadEmbedding(config['eval_char_emb'])
            print 'finish loading embedding'
            encoder = CMNBioCNNEncoder(config, word_emb, char_emb, dropout_p=0)
            decoder = BioRnnDecoder(config, config['encoder_filter_num'], config['hidden_size'],
                                    config['decoder_output_size'], config['output_dim'], 0, config['decoder_layers'])
        else:
            emb = LoadEmbedding(config['eval_emb'])
            # emb = config['loaded_emb']
            print 'finish loading embedding'
            encoder = BioCNNEncoder(config, emb, dropout_p=0)
            decoder = BioRnnDecoder(config, config['encoder_filter_num'], config['hidden_size'],
                                    config['decoder_output_size'], config['output_dim'], 0, config['decoder_layers'])

        # encoder = BioCNNEncoder(config, emb, dropout_p=0)
        # decoder = BioRnnDecoder(config, config['encoder_filter_num'], config['hidden_size'],
        #                         config['decoder_output_size'], config['output_dim'], 0, config['decoder_layers'])
        en_dict = torch.load(os.path.join(config['model_dir'], 'early_encoder_params.pkl'), map_location=lambda storage, loc: storage)
        de_dict = torch.load(os.path.join(config['model_dir'], 'early_decoder_params.pkl'), map_location=lambda storage, loc: storage)
        # en_dict = torch.load('bio_model_eng/early_encoder_params.pkl')
        # de_dict = torch.load('bio_model_eng/early_decoder_params.pkl')
        # en_dict = torch.load('spa_bio_model/encoder_params.pkl')
        # en_dict = {k.partition('module.')[2]: en_dict[k] for k in en_dict}
        # de_dict = {k.partition('module.')[2]: de_dict[k] for k in de_dict}
        # print en_dict.keys()
        encoder.load_state_dict(en_dict)
        decoder.load_state_dict(de_dict)
        # decoder_optimizer = torch.optim.Adadelta(decoder.parameters())
        # encoder_optimizer = torch.optim.Adadelta(encoder.parameters())
        # decoder_optimizer.zero_grad()
        # encoder_optimizer.zero_grad()
        # batch_getter = BioBatchGetter(config, config['dev_data'], 1, shuffle=False)
        if config['USE_CUDA']:
            encoder.cuda(config['cuda_num'])
            decoder.cuda(config['cuda_num'])
        self.encoder = encoder
        self.decoder = decoder

    def batch_process(self, document):
        document._annotate = []
        this_batch = self._batch_data_gen.use_annotaion(document.text_spans)
        top_paths = self.do_batch_process(this_batch)
        for sent_id, top_path in enumerate(top_paths):
            this_batch_len = this_batch[3][sent_id]
            top_path = top_path[1:this_batch_len + 1]
            results = [self._out_vocab.getWord(id) for id in top_path]
            annotate = self._extract_mentions(results, sent_id)
            document._annotate += annotate




    def do_batch_process(self, this_batch):
        top_paths = batch_eva_one_sentence(self._config, self.encoder, self.decoder, this_batch)
        return top_paths





    def process(self, document):
        document._annotate=[]
        for id, sentence in enumerate(document.text_spans):
            tokens= sentence['tokens']
            data = self._data_gen.get_feature(tokens)
            results = self.do_process(data)
            annotate = self._extract_mentions(results, id)
            document._annotate += annotate
    def _extract_mentions(self, seq_results, sent_id):
        annotate=[]
        for i,tag in enumerate(seq_results):
            if tag.startswith('B-'):
                tag= tag[2:]
                mtoks= ['{}_{}'.format(sent_id, i)]
                itag= 'I-' + tag
                for j in range(i+1, len(seq_results)):
                    if seq_results[j] == itag:
                        mtoks.append('{}_{}'.format(sent_id, j))
                    else:
                        break
                mention = dict(mention_tokens=mtoks, md_tag= tag, coref_tag= 'unknown',
                               sent_id=sent_id)
                annotate.append(mention)
        return annotate
                    
        
    def do_process(self, data):
        # rec, costs= self._searcher.search(data, len(data))
        rec = self.infer_eval(data)
        # bestidx = numpy.argsort(costs)[0]
        rec_best = rec  # rec_best= rec[bestidx]
        results = [self._out_vocab.getWord(id) for id in rec_best]
        return results

    def infer_eval(self, data):
        result = eva_one_sentence(self._config, self.encoder, self.decoder, data)
        return result[1:]




        
class NERData(object):
    def __init__(self, config):
        # self.encoding = config['encoding']
        self.config = config
        self._vocabs=config['Vocabs']
        self._outtag_voc = config['BioOutTags']  # self._outtag_voc = config['OutTags']
        # self._tag_pos = config['ner_pos']
        self._word_pos = config['WordPos']
        self._fea_pos= config['fea_pos']
                            
        self._vocab_char= config['CharVoc']
        self._vocab_word = config['WordId']
        self._max_char_len = config['max_char']

        self._use_char_conv=config['use_char_conv']

        self._use_gaz = config['use_gaz']
        if self._use_gaz:
            gazdir=config['GazetteerDir']
            gaz_names= config['Gazetteers']
            self._gazetteers = []
            for (id, gaz) in enumerate(gaz_names):
                gazfile = os.path.join(gazdir, gaz)
                self._gazetteers.append(self._load_gaz_list(gazfile))
    def _load_gaz_list(self, file):
        words=set()
        with codecs.open(file, mode='rb', encoding='utf-8') as f:
            for line in f:
                words.add(line.strip())
        return words
    def get_feature(self, tokens):
        fea_len = len(self._vocabs)
        if self._use_gaz:
            fea_len += len(self._gazetteers)
        if self._use_char_conv:
            fea_len += self._max_char_len*2
        if self.config['lang'] == 'cmn':
            fea_len += 1
        feaMat = numpy.zeros((len(tokens),fea_len),
                             dtype='int32')
        for (lid, token) in enumerate(tokens):
            parts = [token['word_lower'], token['word'],
                     token['caps'], token['pos'],
                     token['ner']]
            if token.has_key('comb-word'):
                parts.append(token['comb-word'])#.encode('utf-8'))
            for (i, voc) in enumerate(self._vocabs):
                fpos= self._fea_pos[i]
                wid = voc.getID(parts[fpos])
                feaMat[lid,i] = wid
            curr_end = len(self._vocabs)
            if self._use_gaz:  
                gazStart = len(self._vocabs)
                for (id, gaz) in enumerate(self._gazetteers):
                    if self.config['lang'] == 'cmn':
                        if parts[5] in gaz:
                            feaMat[lid, id + gazStart] = 1
                    else:
                        if parts[0] in gaz:
                            feaMat[lid, id + gazStart] =1
                curr_end += len(self._gazetteers)
            if self.config['lang'] == 'cmn':
                feaMat[lid, curr_end] = self._vocab_word.getID(parts[5])
                curr_end += 1
            if self._use_char_conv:
                word= parts[self._word_pos]
                chStart = curr_end
                chMaskStart = chStart + self._max_char_len
                for i in range(len(word)):
                    if i >= self._max_char_len:
                        break
                    feaMat[lid, chStart +i] = self._vocab_char.getID(word[i])
                    feaMat[lid, chMaskStart +i] =1
        input_seq_lengths = [feaMat.shape[0]]
        seq_tensor = torch.from_numpy(feaMat[numpy.newaxis, ...]).type(torch.LongTensor)

        return seq_tensor, 1, 1, input_seq_lengths

class CorefData(object):
    def __init__(self,config):
        self._use_char = config['use_char_conv']
        self._use_gaz= config['use_gaz']
        if self._use_gaz:
            raise Exception('gazetter not surpported for coref')
        self._vocab_char = config['CharVoc']
        self._tag_pos = config['corf_pos']
        self._fea_pos = config['fea_pos']
        self._word_pos= config['WordPos']
        self._max_char= config['max_char']
        self._false_anaphore = config['false_anaphore']
        self._false_new = config['false_new']
        self._wrong_link= config['wrong_link']
        
        self._fea_vocs = config['Vocabs']

    def _token_fea(self, tokens):
        fnum = len(self._fea_pos)
        if self._use_char:
            fnum += self._max_char*2
        fea = numpy.zeros((len(tokens),fnum), dtype= 'int32')
        for rid, token in enumerate(tokens):
            parts= [token['word_lower'].encode('utf-8'), token['word'].encode('utf-8'),
                     token['caps'].encode('utf-8'), token['pos'].encode('utf-8'),
                    token['ner'].encode('utf-8')]
            if token.has_key('comb-word'):
                parts.append(token['comb-word'].encode('utf-8'))
            parts.append(token['mention'].encode('utf-8'))
            for id,pos in enumerate(self._fea_pos):
                fea[rid,id] = self._fea_vocs[id].getID(parts[pos])
            if self._use_char:
                word= parts[self._word_pos]
                chStart = len(self._fea_pos)
                chMaskStart = chStart + self._max_char
                for i in range(len(word)):
                    if i >= self._max_char:
                        break
                    fea[ rid, chStart +i] = self._vocab_char.getID(word[i])
                    fea[ rid, chMaskStart +i] =1
        return fea
    def _mention_mask(self, mentions, splits):
        fea_parts = []

        for (start,end, tokens) in splits:
            curr_mentions=[]
            for mention in mentions:
                if mention[0] >= start and mention[0] <end:
                    curr_mentions.append((mention[0]-start, mention[1]-start))
            entity_mask = numpy.zeros((len(curr_mentions), len(tokens)), dtype= 'float32')
            
            for i, mention in enumerate(curr_mentions):
                entity_mask[i, mention[0]:mention[1]] =1
            feaTokens = self._token_fea(tokens)
            feaTokens= feaTokens.reshape((1,feaTokens.shape[0], feaTokens.shape[1]))
            fea_parts.append((feaTokens, entity_mask))
            

        anaphore_mask = numpy.zeros((len(mentions), len(mentions)), dtype= 'float32')
        for i in range(len(mentions)):
            anaphore_mask[i,:i]=1
        return fea_parts,anaphore_mask
                
            
        

    def gen_feature(self, sentences, mentions):
        start = 0
        sent_pos=[]
        for sentence in sentences:
            sent_pos.append( start)
            start += len(sentence['tokens'])
        mention_pos = []
        for mention in mentions:
            sent_id = mention['sent_id']
            tstart = sent_pos[sent_id]
            t0 = int(mention['mention_tokens'][0].split('_')[1]) + tstart
            t1 = int(mention['mention_tokens'][-1].split('_')[1]) + tstart+1
            mention_pos.append((t0,t1))
        doc_parts = self._split_doc(sentences)
        fea_parts,anaphore_mask = self._mention_mask(mention_pos, doc_parts)
        return (fea_parts,anaphore_mask)
    
    def _split_doc(self, sentences, max_len = 2000):
        len_words=sum(len(sent['tokens']) for sent in sentences)
        if len_words < max_len:
            tokens=[]
            for sent in sentences:
                tokens += sent['tokens']
            return [(0, len_words, tokens)]
        part_num= len_words/max_len +1
        len_part = len_words/part_num
        parts= []
        curr_tokens= []
        curr_len = 0
        pre_len = 0
        for sent in sentences:
            curr_tokens += sent['tokens']
            curr_len += len(sent['tokens'])
            
            if curr_len >= len_part:
                parts.append((pre_len, pre_len + curr_len, curr_tokens))
                pre_len = pre_len  + curr_len
                curr_len = 0
                curr_tokens = []
        if len(curr_tokens) >0:
            parts.append((pre_len, pre_len + curr_len, curr_tokens))
        return parts
        
        
            
        
            
