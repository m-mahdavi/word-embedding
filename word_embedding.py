#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
##########################################################


##########################################################
# Word Embedding
# Mohammad Mahdavi
# moh.mahdavi.l@gmail.com
# March 2016
# All Rights Reserved
##########################################################


##########################################################
import os
import re
import gensim
import parameter_configuration as pc
import persian_text_preprocessing as ptp
##########################################################


##########################################################
class WordEmbedding:
    """
    This class implement a word embedding system using Word2Vec.
    """
    LEARNING_METHOD = 1   # CBOW = 0 and Skip-Gram = 1
    VECTOR_SIZE = 200
    WINDOW_SIZE = 8
    NEGATIVE_SAMPLE_COUNT = 10
    ITERATION_COUNT = 20
    MIN_TERM_OCCURRENCE = 30

    def __init__(self):
        """
        This constructor loads model.
        """
        self.ptp = ptp.PersianTextPreprocessing()
        model_backup_path = os.path.join(pc.MODEL_FOLDER, "word_embedding_model.model")
        if os.path.isfile(model_backup_path):
            self.model = gensim.models.Word2Vec.load(model_backup_path)
        else:
            self.makeModel()

    def makeModel(self):
        """
        This method builds a word to vector model.
        """
        print "Loading Corpus..."
        corpus_path = os.path.join(pc.MODEL_FOLDER, "news_corpus.text")
        sentence_list_list = gensim.models.word2vec.LineSentence(corpus_path)
        print "Preparing Corpus for Learning..."
        unigram_list_list = []
        for unigram_list in sentence_list_list:
            sentence_text = self.ptp.normalizer(" ".join(unigram_list))
            sentence_text = re.sub(pc.PUNCTUATION_LIST, " ".decode("utf-8"), sentence_text, flags=re.UNICODE)
            sentence_text = re.sub(pc.SPC, " ".decode("utf-8"), sentence_text, flags=re.UNICODE)
            sentence_unigram_list = [x for x in sentence_text.split(" ") if x != ""]
            unigram_list_list.append(sentence_unigram_list)
        print "Generating Bigrams..."
        bigram = gensim.models.Phrases(unigram_list_list)
        term_list_list = list(bigram[unigram_list_list])
        print "Building Model..."
        self.model = gensim.models.Word2Vec(term_list_list, sg=self.LEARNING_METHOD, size=self.VECTOR_SIZE,
                                            window=self.WINDOW_SIZE, negative=self.NEGATIVE_SAMPLE_COUNT,
                                            iter=self.ITERATION_COUNT, min_count=self.MIN_TERM_OCCURRENCE)
        model_backup_path = os.path.join(pc.MODEL_FOLDER, "word_embedding_model.model")
        self.model.save(model_backup_path)

    def useModel(self, positive_list, negative_list, result_count=20):
        """
        This method uses the word to vector model.
        """
        return self.model.most_similar(positive=positive_list, negative=negative_list, topn=result_count)
##########################################################


##########################################################
if __name__ == "__main__":
    word_to_vector = WordEmbedding()
    # positive_list = ["کرمان".decode("utf-8"),"اردکان".decode("utf-8")]
    # negative_list = ["رفسنجان".decode("utf-8")]
    # positive_list = ["کرمان".decode("utf-8"),"کاشان".decode("utf-8")]
    # negative_list = ["رفسنجان".decode("utf-8")]
    # positive_list = ["برانکو".decode("utf-8"),"استقلال".decode("utf-8")]
    # negative_list = ["پرسپولیس".decode("utf-8")]
    # positive_list = ["تهران".decode("utf-8"),"انگلیس".decode("utf-8")]
    # negative_list = ["ایران".decode("utf-8")]
    # positive_list = ["روحانی".decode("utf-8"),"ترکیه".decode("utf-8")]
    # negative_list = ["ایران".decode("utf-8")]
    # positive_list = ["کی‌روش".decode("utf-8"),"والیبال".decode("utf-8")]
    # negative_list = ["فوتبال".decode("utf-8")]
    # positive_list = ["پایتخت".decode("utf-8"),"ترکیه".decode("utf-8")]
    # negative_list = []
    # positive_list = ["کشور".decode("utf-8"),"میهن".decode("utf-8")]
    # negative_list = []
    # positive_list = ["حربه".decode("utf-8"),"دشمنان".decode("utf-8")]
    # negative_list = []
    # positive_list = ["وزیر".decode("utf-8"),"ورزش".decode("utf-8")]
    # negative_list = []
    # positive_list = ["مالک".decode("utf-8"),"فیسبوک".decode("utf-8")]
    # negative_list = []
    positive_list = ["رئیس".decode("utf-8"), "مجمع_تشخیص".decode("utf-8"), "مصلحت_نظام".decode("utf-8")]
    negative_list = []
    result_count = 20
    result_list = word_to_vector.useModel(positive_list=positive_list, negative_list=negative_list,
                                          result_count=result_count)
    for x, y in result_list:
        print x.replace("_", " ")
##########################################################
