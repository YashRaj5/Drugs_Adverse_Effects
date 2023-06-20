# Databricks notebook source
slides_html="""
<iframe src="https://docs.google.com/presentation/d/e/2PACX-1vSru8d8LC-C_77PM44Wjo7Ti5zR0OkHSUlFU22BRoRhXkIwYIyMhTZ-7AIsf3hCTGcID-Tu2MarTuiT/embed?start=false&loop=false&delayms=3000" frameborder="0" width="960" height="569" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>
"""
displayHTML(slides_html)

# COMMAND ----------

# MAGIC %md
# MAGIC # Detecting Adverse Drug Events From Conversational Texts
# MAGIC
# MAGIC Adverse Drug Events (ADEs) are potentially very dangerous to patients and are top causes of morbidity and mortality. Many ADEs are hard to discover as they happen to certain groups of people in certain conditions and they may take a long time to expose. Healthcare providers conduct clinical trials to discover ADEs before selling the products but normally are limited in numbers. Thus, post-market drug safety monitoring is required to help discover ADEs after the drugs are sold on the market. 
# MAGIC
# MAGIC Less than 5% of ADEs are reported via official channels and the vast majority is described in free-text channels: emails & phone calls to patient support centers, social media posts, sales conversations between clinicians and pharma sales reps, online patient forums, and so on. This requires pharmaceuticals and drug safety groups to monitor and analyze unstructured medical text from a variety of jargons, formats, channels, and languages - with needs for timeliness and scale that require automation. 
# MAGIC
# MAGIC In the solution accelerator, we show how to use Spark NLP's existing models to process conversational text and extract highly specialized ADE and DRUG information, store the data in lakehouse, and analyze the data for various downstream use cases, including:
# MAGIC
# MAGIC - Conversational Texts ADE Classification
# MAGIC - Detecting ADE and Drug Entities From Texts
# MAGIC - Analysis of Drug and ADE Entities
# MAGIC - Finding Drugs and ADEs Have Been Talked Most
# MAGIC - Detecting Most Common Drug-ADE Pairs
# MAGIC - Checking Assertion Status of ADEs
# MAGIC - Relations Between ADEs and Drugs
# MAGIC
# MAGIC There are three notebooks in this package:
# MAGIC
# MAGIC
# MAGIC 1. `./01_ade-extraction`: Extract ADE, DRUGS, assertion status and relationships between drugs and ades
# MAGIC 2. `./02_ade-analysis`: Create a deltalake of ADE and drugs  based on extracted entities and analyze the results (drug/ade correlations)
# MAGIC 3. `./00_config`: Notebook for configuring the environment
# MAGIC
# MAGIC <img src="https://drive.google.com/uc?id=1TL8z5cjKLgXjqCcbgIA4Lfg8M6lXmyzG">
# MAGIC

# COMMAND ----------


