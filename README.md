# Improving Financial Distress Classification System

## Introduction

In normal, non-stressed environment, it is very hard to predict bank's failure as it is a very rare event equivalent of anomaly detection; for more information please visit https://www.bankrate.com/banking/list-of-failed-banks/. 

There was a significant increase in the number of failed banks in the US from 2009 to 2014 what produced enough data for effective classification. Notwithstanding the spike in failures, it was still necessary to create comparable risk profiles. Below are annual counts of regulated banks, healthy in blue and failed in red.

![](assets/all_banks.png) 

The primary objective was to develop an early warning system, i.e. binary classification of failed (`'Target'==1`) vs. survived (`'Target'==0`), for the US banks using their quarterly filings with the regulator. Overall, 137 failed banks and 6,877 surviving banks were used in this machine learning exercise. Historical observations from the first 4 quarters ending 2010Q3 (stored in `./data`) are used to tune the model and out-of-sample testing is performed on quarterly data starting from 2010Q4 (stored in `./oos`).  For more information on methodology please refer to supplemental `CAMELS.md` file included in the repository. Below are annual failures showing a clear increase in counts.

![](assets/failed_banks.png)

## Dataset

### Overview

Approximately 2,000 original features were obtained for every bank instance from "Report of Condition and Income" (CALL report) using publicly available APIs at https://banks.data.fdic.gov/docs/. Sample report is available here `'data/CALL_175458.PDF'`. Eventually, only 14 financial metrics were used for the actual classification:

![](assets/selected_financials.png)

For more information about CALL reports please visit the following resources:

-   regulator's website at https://cdr.ffiec.gov/public/ManageFacsimiles.aspx
-   detailed description is also available here https://www.investopedia.com/terms/c/callreport.asp

### Task

Selected financial ratios were used to produce comparable risk profiles according to CAMELS valuation framework, that is explained in detail in supplemental `CAMELS.md` file. For more information about CAMELS framework please visit the following resources:

 -   regulator's website at https://www.fdic.gov/deposit/insurance/assessments/risk.html 
 -   datailed explanation is also available here https://en.wikipedia.org/wiki/CAMELS_rating_system.

This framework can be used to assess performance along six risk dimensions: 1) Capital, 2) Assets, 3) Management, 4) Earnings, 5) Liquidity, and 6) Sensitivity to market risk. It was assumed that a failed bank will exceed its risk capacity, i. e. hypothetical outside contour, along several dimensions and eventually would face a liquidity crises. 

![](assets/single_CAMELS.png)

Financial metrics recorded in the last reports of the failed banks should have predictive power that is needed to forecast future failures. Due to significant class imbalances and taking into account costs associated with financial distress, the model should aim to maximize the recall score. In other words, accuracy is probably not the best metrics, as Type II error needs to be minimized.

Basic benchmark model was created in order to better understand the requirements. Sklearn `train-test-split` was used with `StandardScaler` to prepare for Gradient Boosting tree-based `GridSearch`, optimizing for recall. The trained model performed reasonably well on the testing dataset with AUC of 0.97. Out-of-sample results were also very promising as recall scores  were ranging from 0.76 to 1. Out of 138 banks that failed during the period from 2010Q4 to 2012Q4 **the benchmark model correctly flags 124 failed banks out of 138 true failed** based solely on the information from their last CALL reports. With time the number of failed banks decreases sharply and so does predictive power of the model.

![](assets/oos_GBM.png)

### Access

Training data with respective CAMELS features was stored in CSV format `'data/camel_data_after2010Q3.csv'`. GUI was used to uploaded the training dataset to Azure storage to be accessed with `dataset = ws.datasets['camels']`. Two columns with bank's id and report's date are excluded from the schema. Additionally, `train.py` script has direct access to the training dataset staged in raw on GitHub. Out-of-sample testing data is stored in `.oos/` folder and CSV files have `'_OOS.csv'` tag at the end of file name.

The data from last reports of failed banks is collected for 4 preceding quarters prior to 2010Q3 as noted in `'AsOfDate'` column. All healthy or surviving banks are taken as of 2010Q3 quarter-end. This means that the failed banks are represented by a panel dataset with observations from different quarters. For example, if failed Bank A submitted its last report in 2010Q1 and another failed Bank B submitted its last report in 2010Q2, both banks will appear in the training dataset. Healthy Bank C could have submitted reports in Q1 and Q2 as well, but only features as of 2010Q3 will be used for training, as it survived after 2010Q3 reporting cycle. This important choice was necessary in order to partially mitigate imbalanced classes.

Out-of-sample testing data covers 9 quarters starting in 2010Q4 and contains reports submitted by failing and surviving banks as of respective quarter-end. Obviously, failed banks will not appear in the next quarterly reporting cycle, but not all missing banks are automatically recorded as failed banks. There are numerous reasons why reports can disappear from the sample, for example consolidation, change of charter or mergers and acquisitions. These out-of-sample quarterly datasets are stored in respective CSV files, for example `'oos/camel_data_after2010Q4_OOS.csv'` contains CAMELS features reported by all existing banks in Q4 of 2010.


## Hyperparameter Tuning

### Methodology

Similarly to the benchmark model, Gradient Boosting Classifier was chosen as it seems to be a very flexible algorithm for classification tasks. Depending on hyperparamenters, it can mimic high-bias AdaBoosting as well as high-variance RandonForest. Moreover, it provides feature importances and predicts probabilities, similar to tree-based classifiers and logistic regression.

### Experiment Setup

Following the benchmark approach: the list of hyperparameters to tune included the following: learning rate, number of estimates, maximum number of features,  and maximum depth of a tree.

        {'learning_rate': 0.1, 'max_depth': 2, 'max_features': 5, 'n_estimators': 20}

It was interesting to see if HyperDrive could possibly recommend radically different settings. Therefore, Bayesian sampling was selected with the same random_state=123 for the actual classification model. The choice of parameter sampling required at least 80 runs.

![](assets/hdr_config.png)

It is important to point out that AutoML can't to optimize basic recall score directly. Available primary metric, that seems to be close to the recall score, is called norm_macro_recall from 0, random, to 1, perfect performance. Alternatively, AUC_weighted was considered as it considers an arithmetic mean of the score for each class, weighted by the number of true instances in each class. This optimization should probably be benchmarked agains sklearn grid search for roc_auc. (Source https://github.com/MicrosoftDocs/azure-docs/blob/master/articles/machine-learning/how-to-understand-automated-ml.md) 

### Results

HyperDrive run was completed in approximately 1 hour and 50 minutes and produced promising results. The best model produced norm_macro_recall of 0.74:

         {'Learning rate:': 0.1,
         'Number of estimators:': 20,
         'Number of features:': 3,
         'Max tree depth:': 2,
         'norm_macro_recall': 0.74286}

Further improvements could include performing 4 or more concurrent runs, as well as removing data wrangling functionality from the training script. Below are screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

![](assets/hdr_rundetails1.png)
![](assets/hdr_rundetails2.png)
![](assets/hdr_best_model_id.png)
![](assets/hdr_bestrun_id.png)
![](assets/hdr_best_performance.png)

Although the best run had good performance in terms of recall, out-of-sample performance did not noticable differ from the benchmark model.
![](assets/hdr_oos_performance.png)

## Automated ML

### Methodology

Generally speaking, decision trees should work well for this task, as these models do not make any functional form assumptions, handle both categorical and continuous data well, and are easy to interpret. Tree-based models simply aim to reduce entropy at every split and are therefore very straightforward, no need to worry about missing data and scaling. They are not very stable though, as new data may produce a totally different tree, and they also tend to overfit.

Possible solution would be model averaging - employing “wisdom of the crowd”. It seems that for the present task two paths are possible: reducing variance or reducing bias. The former implies complex model, i.e. starting with a bushy, high-variance tree and resampling with replacement, what will produce a family of Random Forest models. The later implies starting with a simple model, i.e. possible a stump, high-bias classifier and learning from miss-classified instances, what will produce a family of Boosting models.

### Experiment Setup

For the experiments in this section it was decided to run from workspace blob storage; obviously the same dataset:
![](assets/camels11_dataset.png)

As discussed above, achieving good recall score was the main goal and this is why `'norm_macro_recall'` was chosen as a primary metric. Timeout and number of concurrent iterations were set conservatively to control the costs.
![](assets/aml_config.png)

Below are run details followed by the best model.
![](assets/aml_rundetails1.png)
![](assets/aml_rundetails2.png)

The best model, `VotingEnsemble`, was registered
![](assets/aml_register_model.png)
run ID is presented here:
![](assets/aml_best_runid.png)

### Results

Automated machine learning performed exceptionally well producing a number of outstanding recall scores. Performance of the model was analysed in terms of its classification power:
![](assets/aml_perf_metrics.png)
![](assets/aml_confusion.png)
![](assets/aml_features.png)
![](assets/aml_precision_recal.png)
![](assets/aml_auc.png)

The model was tested on 9 out-of-sample dataset and generally performed better than Gradient Boosting classifier in terms of recall. Precision score was also low as expected. This **`VotingEnsemble` model is able to flag 135 failed banks out of 138 true failed, as compared to only 124 flagged by Gradient Boosting**.
![](assets/aml_oos_performance.png)

Further improvements could include optimizing for AUC and training/testing on more data, say from 2008 to 2016. Also, it would be very interesting to explore 6 missed banks in terms of fundamental analysis.

## Model Deployment

The `VotingEnsemble` model tuned using automated machine learning achieved `norm-macro-recall` of 0.94 and was selected for deployment. The model `automl_model.pkl` was saved and registered first. Successful deployment was confirmed here:
![](assets/aciservice_created.png)
![](assets/aciservice_healthy.png)
![](assets/aciservice_keys.png)
![](assets/aciservice_deployment_logs.png)

Endpoints were created here:
![](assets/aci_endpoints.png)

Testing for Positive and Negative instances was performed; details are in automatically generated `scoring.py` file. Here is an example of JSON payload with CAMELS features for 5 failed banks followed by responce [1, 1, 1, 1, 1], i.e. all are correctly predicted to fail.
![](assets/aci_request.png)
![](assets/aci_response.png)

Here is an example of the service logs:
![](assets/aci_service_logs.png)

## Screen Recording

Link to a screen recording of the project in action: https://www.youtube.com/watch?v=UCfJ44DDScY

- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

