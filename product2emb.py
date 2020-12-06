#!/usr/bin/env python
# coding: utf-8
from common import data, date, utils
from common.log import log

from model import base
from gensim.models import Word2Vec
import json
import os


def get_data():
    sql = """
            select uid as user_id, create_time, product_mongo_id
            from ddmc_recommend.user_action
            where dt >= date_sub(to_date(now()), 7)
              and action_type = 'product_detail'
            """
    d = data.get_impala_data_df(sql)
    return d


def training_process():
    log.info("start training product model...")
    df_raw = get_data()
    visit_session = base.build_session(df_raw, min_n=5, gap_thre=3600 * 3)  # 创建会话
    model = Word2Vec(sentences=visit_session, window=5, sg=0, size=100, workers=10)
    return model


def get_convert_dict():
    file_name = str(date.get_yesterday()) + 'mongoid2name.json'
    mongoid2name = None
    if os.path.exists(file_name):
        with open(file_name) as f:
            mongoid2name = json.load(f)
    if mongoid2name is not None:
        return mongoid2name

    sql = """
        SELECT product_id, product_mongo_id, product_name 
        from dim.product_hive 
        WHERE snapshot = date_sub(to_date(now()), 1)
        """
    df_product = data.get_impala_data_df(sql)

    mongoid2name = {k: v for k, v in df_product[['product_mongo_id', 'product_name']].values}
    with open(file_name, 'w') as f:
        json.dump(mongoid2name, f)
    # name2mongoid = {k: v for v, k in df_product[['product_mongo_id', 'product_name']].values}
    # mongoid2id = {k: v for k, v in df_product[['product_mongo_id', 'product_id']].values}
    return mongoid2name


def get_similar_output(model):
    similar_dict = {}
    words = list(model.wv.vocab.keys())
    mongoid2name = get_convert_dict()
    for w in words:
        sim = model.wv.most_similar(w, topn=50)
        sim_list = [{'mongo_id': s[0], 'p': s[1], 'p_name': mongoid2name.get(s[0])} for s in sim]
        similar_dict[w] = sim_list
    return similar_dict


def test_model(model_path, pid):
    model = Word2Vec.load(model_path)
    sim = model.wv.most_similar(pid, topn=50)
    mongoid2name = get_convert_dict()
    for i in sim:
        print(i, mongoid2name.get(i[0]))


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        print("%s <model_path> <pid>" % sys.argv[0])
        exit(1)

    test_model(sys.argv[1], sys.argv[2])
