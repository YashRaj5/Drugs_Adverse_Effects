# Databricks notebook source
# MAGIC %md
# MAGIC # Extracting Entities and Realtionships
# MAGIC In this note book we use JohnSNow Lab's [pipelines for adverse drug events](https://nlp.johnsnowlabs.com/2021/07/15/explain_clinical_doc_ade_en.html) to extract adverse events (ADE) and drug entites from a collection of 20,000 conversational texts. We then store the ectracted entities and raw data in delta lake and analyze the data in later notebook.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Initial Configuration

# COMMAND ----------

import json
import os
 
from pyspark.ml import PipelineModel,Pipeline
from pyspark.sql import functions as F
from pyspark.sql.types import *
 
from sparknlp.annotator import *
from sparknlp_jsl.annotator import *
from sparknlp.base import *
import sparknlp_jsl
import sparknlp
 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
 
warnings.filterwarnings("ignore")
pd.set_option("display.max_colwidth",100)
 
print('sparknlp.version : ',sparknlp.version())
print('sparknlp_jsl.version : ',sparknlp_jsl.version())
 
spark

# COMMAND ----------

# MAGIC %md
# MAGIC For simplicity and ease of use, we run a notebook in the collection which contains definitions and classes for setting and creating paths as well as downloading relevant datasets for this exercise.
# MAGIC

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# DBTITLE 1,initial configurations
ade_demo_util=SolAccUtil('ade')
ade_demo_util.print_paths()
ade_demo_util.display_data()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download Dataset
# MAGIC
# MAGIC We will use a slightly modified version of some conversational ADE texts which are downloaded from https://sites.google.com/site/adecorpus/home/document. See
# MAGIC >[Development of a benchmark corpus to support the automatic extraction of drug-related adverse effects from medical case reports](https://www.sciencedirect.com/science/article/pii/S1532046412000615)
# MAGIC for more information about this dataset.
# MAGIC
# MAGIC **We will work with two main files in the dataset:**
# MAGIC
# MAGIC - DRUG-AE.rel : Conversations with ADE.
# MAGIC - ADE-NEG.txt : Conversations with no ADE.
# MAGIC
# MAGIC Lets get started with downloading these files.

# COMMAND ----------

# DBTITLE 1,download dataset
for file in ['DRUG-AE.rel','ADE-NEG.txt']:
  try:
    dbutils.fs.ls(f'{ade_demo_util.data_path}/{file}')
    print(f'{file} is already downloaded')
  except:
    ade_demo_util.load_remote_data(f'https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/tutorials/Certification_Trainings/Healthcare/data/ADE_Corpus_V2/{file}')
    
ade_demo_util.display_data()

# COMMAND ----------

# MAGIC %md
# MAGIC Now we will create a dataset of texts with the ground-truth classification, with respect to their ADE status.

# COMMAND ----------

# DBTITLE 1,dataframe for negative ADE texts
neg_df = (
  spark.read.text(f"{ade_demo_util.data_path}/ADE-NEG.txt")
  .selectExpr("split(value,'NEG')[1] as text","1!=1 as is_ADE")
  .drop_duplicates()
)
display(neg_df.limit(20))

# COMMAND ----------

# DBTITLE 1,dataframe for positive ADE texts
pos_df = (
  spark.read.csv(f"{ade_demo_util.data_path}/DRUG-AE.rel", sep="|", header=None)
  .selectExpr("_c1 as text", "1==1 as is_ADE")
  .drop_duplicates()
)
 
display(pos_df.limit(20))

# COMMAND ----------

# DBTITLE 1,dataframe for all conversational texts with labels
raw_data_df = neg_df.union(pos_df).selectExpr('uuid() as id', '*').orderBy('id')
raw_data_df.display()

# COMMAND ----------

raw_data_df.groupBy('is_ADE').count().display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## write ade_events to Delta
# MAGIC We will combine the two dataframes and store the data in the bronze delta layer

# COMMAND ----------

raw_data_df.repartition(12).write.format('delta').mode('overwrite').save(f'{ade_demo_util.delta_path}/bronze/ade_events')

# COMMAND ----------

# MAGIC %md 
# MAGIC # Conversational ADE Classification

# COMMAND ----------

# MAGIC %md
# MAGIC ## Use Case: Text Classification According to Contains ADE or Not

# COMMAND ----------

# MAGIC %md
# MAGIC Now we will try to predict if a text contains ADE or not by using `classifierdl_ade_conversational_biobert`. For this, we will create a new data frame merging all ADE negative and ADE positive texts and shuffle that

# COMMAND ----------

ade_events_df=spark.read.load(f'{ade_demo_util.delta_path}/bronze/ade_events').orderBy(F.rand(seed=42)).repartition(64).cache()
display(ade_events_df.limit(20))

# COMMAND ----------

document_assembler = DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")
 
tokenizer = Tokenizer()\
        .setInputCols(['document'])\
        .setOutputCol('token')
 
embeddings = BertEmbeddings.pretrained('biobert_pubmed_base_cased')\
        .setInputCols(["document", 'token'])\
        .setOutputCol("embeddings")
 
sentence_embeddings = SentenceEmbeddings() \
        .setInputCols(["document", "embeddings"]) \
        .setOutputCol("sentence_embeddings") \
        .setPoolingStrategy("AVERAGE")
 
conv_classifier = ClassifierDLModel.pretrained('classifierdl_ade_conversational_biobert', 'en', 'clinical/models')\
        .setInputCols(['sentence_embeddings'])\
        .setOutputCol('conv_class')
 
clf_pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer, 
    embeddings, 
    sentence_embeddings, 
    conv_classifier])
 
empty_data = spark.createDataFrame([['']]).toDF("text")
clf_model = clf_pipeline.fit(empty_data)

# COMMAND ----------

# MAGIC %md Now we transform the conversational texts dataframe using the ADE classifier pipeline and write the resulting dataframe to delta lake for future access.

# COMMAND ----------

clf_model_results_df=clf_model.transform(ade_events_df).select("id", "text", "is_ADE" ,"conv_class.result")
clf_model_results_df.write.format('delta').mode('overwrite').save(f'{ade_demo_util.delta_path}/silver/clf_model_results')

# COMMAND ----------

clf_model_results_df=(
  spark.read.load(f'{ade_demo_util.delta_path}/silver/clf_model_results')
  .selectExpr("id", "text", "is_ADE" ,"cast(result[0] AS BOOLEAN) as predicted_ADE")
)

# COMMAND ----------

display(clf_model_results_df)

# COMMAND ----------

clf_pdf=clf_model_results_df.selectExpr('cast(is_ADE as int) as ADE_actual', 'cast(predicted_ADE as int) as ADE_predicted').toPandas()
confusion_matrix = pd.crosstab(clf_pdf['ADE_actual'], clf_pdf['ADE_predicted'], rownames=['Actual'], colnames=['Predicted'])
confusion_matrix

# COMMAND ----------

# MAGIC %md
# MAGIC Based on the above confusion matrix our model has an accuracy of `(TP+TN)/(P+N) = 86%`

# COMMAND ----------

(3249+14753)/(1022+3249+14753+1872)

# COMMAND ----------

# MAGIC %md
# MAGIC # ADE-DRUG NET Examination
# MAGIC Now we will extract `ADE` and `DRUG` entities from the conversational texts by using combination of `ner_ade_clinical` and `ner_posology` models.

# COMMAND ----------

ade_df = spark.read.format('delta').load(f'{ade_demo_util.delta_path}/bronze/ade_events').drop('is_ADE').repartition(64)
display(ade_df.limit(20))

# COMMAND ----------

ade_df.count()

# COMMAND ----------

# DBTITLE 1,create pipeline
documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")
 
sentenceDetector = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")
 
tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")
 
word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")
  
ade_ner = MedicalNerModel.pretrained("ner_ade_clinical", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ade_ner")
 
ade_ner_converter = NerConverter() \
    .setInputCols(["sentence", "token", "ade_ner"]) \
    .setOutputCol("ade_ner_chunk")\
 
pos_ner = MedicalNerModel.pretrained("ner_posology", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("pos_ner")
 
pos_ner_converter = NerConverter() \
    .setInputCols(["sentence", "token", "pos_ner"]) \
    .setOutputCol("pos_ner_chunk")\
    .setWhiteList(["DRUG"])
 
chunk_merger = ChunkMergeApproach()\
    .setInputCols("ade_ner_chunk","pos_ner_chunk")\
    .setOutputCol("ner_chunk")\
 
 
ner_pipeline = Pipeline(stages=[
    documentAssembler, 
    sentenceDetector,
    tokenizer,
    word_embeddings,
    ade_ner,
    ade_ner_converter,
    pos_ner,
    pos_ner_converter,
    chunk_merger
    ])
 
 


# COMMAND ----------

empty_data = spark.createDataFrame([[""]]).toDF("text")
ade_ner_model = ner_pipeline.fit(empty_data)

# COMMAND ----------

# DBTITLE 1,transform dataframe of texts
ade_ner_result_df = ade_ner_model.transform(ade_df)

# COMMAND ----------


