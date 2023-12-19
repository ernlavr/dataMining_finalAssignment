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
2. Run `python3 main.py --args_path conf/args.yaml`
3. 