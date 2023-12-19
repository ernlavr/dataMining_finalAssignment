# dataMining_finalAssignment

## Data
Download CRESCI-2015 https://botometer.osome.iu.edu/bot-repository/datasets.html

Add TWT and E13 into the data folder

```
data/e13/*.csv;
data/twt/*.csv;
```

## Run

### Generate preprocessed data and perform clustering
1. Install requirements `pip install -r requirements.txt`
2. Ensure that you have the data in the correct folder, see above
3. Run `python3 main.py --args_path conf/args.yaml`
4. The output will be saved in `output/` for each corresponding folder
5. Additionally you can also view notebooks in `src/ml/cl.ipynb` and `src/ml/sl.ipynb`
for clustering and supervised learning respectively. They contain minimum code
to run the models and observe the results.