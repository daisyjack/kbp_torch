# -*- coding: utf-8 -*-
from __future__ import absolute_import
from .document import ParsedDocument,XmlDocument
from .corenlp import StanfordCoreNLP
from .annotation import KBPAnnMgr,ApfAnnMgr
import os
import codecs
import sys
def _list_files(path, expand):
    files= []
    if not os.path.isdir(path):
        print '{} is not directory'.format(path)
        return files
    try:
        items = os.listdir(path)
        for item in items:
            fname = os.path.join(path, item)
            if os.path.isdir(fname):
                files += _list_files(fname, expand)
            elif os.path.isfile(fname):
                if fname.endswith(expand):
                    files.append(fname)
    except Exception, message:
        print message
    return files
def _mention_span(mtoks, tspans):
    begin = mtoks[0].split('_')
    (s_begin, t_begin)= (int(begin[0]), int(begin[1]))
    char_begin = tspans[s_begin][u'tokens'][t_begin][u'char_begin']
    end =  mtoks[-1].split('_')
    (s_end, t_end)= (int(end[0]), int(end[1]))
    char_end = tspans[s_end][u'tokens'][t_end][u'char_end']
    text= u''
    for mtok in mtoks:
        mspan= mtok.split('_')
        (s,t)= (int(mspan[0]), int(mspan[1]))
        tok = tspans[s][u'tokens'][t]
        text += tok['origin_text'] +u' '
    text=text.strip()
        
    return (char_begin, char_end,text)
def _is_overlap(s1,e1, s2,e2):
    return max(s1,s2) < min(e1,e2)    
    
def _check_doc(annotation, doc):
    for mg in annotation.mentions:
        (start,end, enttype, coref,mtext) = (mg._char_begin, mg._char_end,
                                       '{}_{}'.format(mg._entity_type, mg._mention_type),
                                       mg._coref_id, mg._mention)
        is_checked= False
        for ann_text in doc._annotate:
            mtoks= ann_text['mention_tokens']
            ent_doc= ann_text['md_tag']
            coref_doc= ann_text['coref_tag']
            if len(mtoks)==0:
                sys.stderr.write('{} has zero token mention'.format(doc._name))
                continue
            (char_begin,char_end,dtext) = _mention_span(mtoks, doc._text_spans)
            if (char_begin, char_end, ent_doc, coref_doc)==(start, end, enttype, coref):                
                is_checked= True
                break
        if is_checked:
            continue
        for ann_text in doc._annotate:
            mtoks= ann_text['mention_tokens']
            ent_doc= ann_text['md_tag']
            coref_doc= ann_text['coref_tag']
            if len(mtoks)==0:
                sys.stderr.write('{} has zero token mention'.format(doc._name))
                continue
            (char_begin,char_end,dtext) = _mention_span(mtoks, doc._text_spans)
            if _is_overlap(char_begin, char_end, start, end):
                if not (char_begin, char_end)==(start, end):
                    try:
                        
                        sys.stdout.write(u'{}\t{}:{}\t{}:{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                            doc._name, start,end-1, char_begin, char_end-1, enttype, ent_doc,
                            coref, coref_doc, mtext,dtext
                            ).encode('utf-8'))
                    except Exception,err:
                        print err
                        raise err
                is_checked= True
                break
        if not is_checked:
            for ann_tag in doc._tag_annotate:
                char_begin= ann_tag['char_begin']
                char_end = ann_tag['char_end']
                tag_id=  ann_tag['tag_id']
                if (char_begin,char_end)==(start,end):
                    is_checked = True
                    if tag_id<0:
                        sys.stderr.write('{} in_tag mention error {}:{}\n'.format(
                            doc._name, char_begin, char_end))
        if not is_checked:
            sys.stderr.write('{} {}:{} not in doc'.format(
                doc._name, start, end))
    
                    
            
            
                                 
            
        

def check_jsons(jsondir, tabfile,config):
    files = _list_files(jsondir, '.json')
    golden = KBPAnnMgr(tabfile, col_title=config['col_title'], 
                       col_link=config['col_link'], col_ner= config['col_ner'], 
                       col_mt= config['col_mt'])
    for file in files:
        doc = ParsedDocument()
        doc.load(file)
        annotation = golden.get_annotation(doc._name)
        if annotation is None:
            if len(doc._annotate)+ len(doc._tag_annotate)>0:
                sys.stderr.write('tags 0 vs {} in {}'.format(
                    len(doc._annotate)+ len(doc._tag_annotate), doc._name))
            continue
        _check_doc(annotation,doc)
    
        
            
  
def _load_quote_regions(quotefile):
    fin = open (quotefile)
    regions=dict()
    for line in fin:
        parts= line.strip().split()
        if len(parts) < 3:
            continue
        if not regions.has_key(parts[0]):
            regions[parts[0]] = []
        regions[parts[0]].append((int(parts[1]), int(parts[2])+1))
        
    return regions


def process_kbp_data(config):
    data_dir = config['data_dir']
    out_dir = config['json_dir']
    quotefile = config['quote_regions']
    tabfile = config['golden_tab']
    expand = config['expand']
    
    corenlp = StanfordCoreNLP(lang=config['language'])
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    files = _list_files(data_dir, expand)
    quote_regions= _load_quote_regions(quotefile)
    golden = None
    if tabfile:
        golden = KBPAnnMgr(tabfile, col_title=config['col_title'], 
                           col_link=config['col_link'], col_ner= config['col_ner'], 
                           col_mt= config['col_mt'])    
    for file in files:
        doc = XmlDocument(ignore_tag_len=config['ignore_tag_len'], ignore_first_line=config['ignore_first_line'], fname = file)
        doc.ignore_spans(quote_regions.get(doc._name))
        doc = ParsedDocument(doc)
        doc.parse_document(corenlp)
        doc.split_sentence()
      
        if golden is not None:
            annotation = golden.get_annotation(doc._name)
            doc.attach_annotation(annotation)
        oname = os.path.join(out_dir, doc._name+'.json')
        doc.dump(oname)
        print '{} processed'.format(doc._name)

def _sequence_tag(doc):
    outlines = '#begin document ({}):\n'.format(doc._name)
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
        for token,mention,coref in zip(tokens,mention_tags, coref_tags):
            line = u'{} {} {} {} {}'.format(token['word_lower'], token['word'],token['caps'], token['pos'], token['ner'])
            if token.has_key(u'comb-word'):
                line += ' '+ token[u'comb-word']
            line+= u' {} {} {} {}\n'.format(mention, coref, token['char_begin'], token['char_end'])
            outlines += line
        outlines += '\n'
    outlines += '#end document\n\n'
    return outlines
                
            
            
    

#this will ignore all nested tags
def gen_sequence_tags(json_dir, outfile, encoding = 'utf-8'):
    fout= codecs.open(outfile, 'w', encoding= encoding)
    files = _list_files(json_dir, '.json')
    for file in files:
        doc = ParsedDocument()
        doc.load(file)
        outlines = _sequence_tag(doc)
        fout.write(outlines)
        fout.flush()
    fout.close()

    
        
        
    

        
            
            
        
