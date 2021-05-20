from flask import render_template, json, jsonify, request, abort, g
from api import app
import torch
from pre import predict

LABEL2ID = {'B':0, 'I':1, 'O':2, 'X':3, '[start]':4, '[end]':5}
CONFIG_FILE = 'config.json'
PRETRAIN_MODEL_FILE = 'saved_models/biobert-base-cased-v1.1.pth'
MODEL_FILE = 'saved_models/disease_model.pt'
DENSE_MODEL_FILE = 'saved_models/to_embedding.pt'
RE_MODEL_FILE = 'saved_models/model_best.pt'

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
NER_MODEL = [
    {
        'type_id': 0,
        'model_name': 'saved_models/ner_model/disease_model.pt', 
        'entity_type': 'disease'
    },
    {
        'type_id': 1,
        'model_name': 'saved_models/ner_model/gene_model.pt', 
        'entity_type': 'gene'
    },
    {
        'type_id': 2,
        'model_name': 'saved_models/ner_model/tissue_model.pt',
        'entity_type': 'tissue'
    }
]
    

@app.route('/predict', methods=['POST'])
def ner():
    data = request.get_json()
    res = []
    if data.get('contentType') == 'PMIDS':
        for item in data['data']:
            tmp = predict(item['sents'], CONFIG_FILE, NER_MODEL,
                            PRETRAIN_MODEL_FILE, DENSE_MODEL_FILE,
                            RE_MODEL_FILE, device=DEVICE)
            res.append({
                'PMID': item.get('PMID', 0),
                'sents': tmp
            })
    elif data.get('contentType') == 'text':
        tmp = predict(data['data']['sents'], CONFIG_FILE, NER_MODEL,
                        PRETRAIN_MODEL_FILE, DENSE_MODEL_FILE,
                        RE_MODEL_FILE, device=DEVICE)
        res.append({
            "sents": tmp
        })
    else:
        return jsonify({"success": False, "code":400003, "message": "无法解析内容"})
    return json.dumps(res)
