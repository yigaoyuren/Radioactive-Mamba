# Radioactive-Mamba
Please wait for a moment. I'll upload the code shortly.
## Data
The dataset used in the paper can be found here：

ZhangLab：https://drive.google.com/file/d/1hOUIoPQlFcnY8f3ZdgOK7By7nRiTvNkX/view?usp=drive_link

Chexpert：https://drive.google.com/file/d/1NBwSfCI526ofT1IhFc-P3BvMqj-l1PMA/view?usp=drive_link

COVIDx：https://drive.google.com/file/d/1wM9nkMQPW2DJ7DDkfaRu-qmQYBtpr7_-/view?usp=drive_link

Change the address of your dataset in configs/base.py
## Train
``` 
python3 main.py --exp zhang_exp --config zhang_best
```
## Test
```
python3 eval.py --exp zhang_exp
