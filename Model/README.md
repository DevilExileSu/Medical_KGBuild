# pytorch-template

pytorch使用模板，简化并规范模块编写。

```bash
├── api                             
│   ├── views.py                    # 向外提供API接口
│   └── __init__.py                 # Flask初始配置
├── config
│   ├── cfg.py                      # 配置类， 配置文件的保存和加载
│   ├── __init__.py 			
│   └── logger.py                   # 	日志类， 打印并保存训练中相关信息以及训练结果
├── config.json                     # 	配置文件， 每个可控参数的设置
├── data                            # 数据加载模块
│   ├── dataloader.py               # 	不使用torch.nn.data
│   ├── dataset.py                  # 	借助torch.nn.data加载以及处理数据集
│   └── __init__.py
├── dataset                         # 模型所使用的训练集	
│   ├── ner                         #   存放命名实体识别数据
│   ├── re                          #   存放远程监督关系抽取数据
│   ├── to_embedding                #   训练降维层所使用数据
│   └── vocab.txt                   #   BERT所使用的字典
├── logs                            # 日志
│   ├── debug.log                   # 	debug日志文件: 模型调试信息 logger.debug("....")
│   └── info.log                    # 	info日志文件： 模型训练损失，评估信息 logger.info("...")
├── model                           # 该模块存放网络模型结构
│   ├── attention.py                #   多头注意力层实现
│   ├── base_model.py               # 	模型基类
│   ├── bert.py                     #   BERT模型总体实现
│   ├── crf_ner.py                  #   使用CRF的命名实体识别模型
│   ├── embedding.py                #   BERT输入
│   ├── encoder.py                  #   Transformer、BERT的Encoder块
│   ├── feedforward.py              #   前向反馈层，先升维再降维
│   ├── pcnn.py                     #   PCNN+ATT+MIL 实现
│   └── pretrain_model.py           #   BERT预训练模型
├── saved_models                    # 保存模型
├── trainer                         # 该模块存放模型训练器
│   ├── embedding_trainer.py        #   降维层训练器
│   ├── mil_trainer.py              #   远程监督关系抽取训练器
│   ├── ner_trainer.py              #   命名实体识别训练器
│   └── trainer.py                  # 	训练器基类
├── utils                           # 通用的函数和类模块
│   ├── metrics.py                  # 	评估函数
│   ├── tokenizer.py                #   BERT分词器
│   └── util.py                     # 	工具函数
├── embedding_train.py              #   降维层训练
├── mil_train.py                    #   关系抽取训练
├── pre.py                          #   关系抽取和实体抽取预测函数
├── run.py                          #   开启模型端服务
├── train.py                        #   ner训练

```

### 例子

[DevilExileSu/transformer](https://github.com/DevilExileSu/transformer) 

[DevilExileSu/AGGCN](https://github.com/DevilExileSu/AGGCN)
