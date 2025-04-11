## Prepare Spider Data

Download [spider data](https://drive.google.com/uc?export=download&id=1TqleXec_OykOYFREKKtschzY29dUcVAQ) and database (only spider original database right now) and then unzip them:

```shell
mkdir data 
unzip spider.zip 
mv spider/database . 
mv spider data
```

Clone evaluation scripts (test-suite-sql-eval: [https://github.com/taoyds/test-suite-sql-eval](https://github.com/taoyds/test-suite-sql-eval)): 

```shell
mkdir datasets_directory/Spider/third_party
cd datasets_directory/Spider/third_party
git clone https://github.com/taoyds/test-suite-sql-eval
cd ..
```

(We referenced https://github.com/bigbigwatermalon/C3SQL/tree/master for the instructions to setup the Spider dataset)

## Verify Batteries Included Data

GSM8k, MMLU, and PretrainDatasets are battery included datasets in this repository as the dataset size is small. They do not require any further setup and the data can be found in `datasets_directory/DATASET_NAME/data` where DATASET_NAME is the dataset you are interested in finding. 

## Create Custom Data For Conditioning

Create a new directory in `datasets_directory` for your dataset and reference `datasets_directory/datasets_template.py` or other files to create your dataset. `datasets_directory/datasets_template.py` defines all the relevant functions that are required for defining a new dataset, and we attempt to put all files relevant to a specific dataset into its own directory for easy reference. 

After creating your `DATASET_NAME_utils.py` file, `utils/measurement_utils.py` and `gptq/datautils.py` should be edited to support your new dataset, this should be an intuitive edit as long as your dataset has been implemented following the same argument conventions as previous datasets. 