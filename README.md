# Try-On Module
Building a *Character Preserving Virtual Try-On System* using deep learning. The project is initiated as 
college project work.


### Resources

- [Data source](https://drive.google.com/file/d/1MxCUvKxejnwWnoZ-KoCyMCXo3TLhRuTo/view)
- [CP-VITON github](https://github.com/sergeywong/cp-vton)


## Enviroment Setup

```sh
git clone https://github.com/rosxnb/viton.git
cd viton
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


**To run model on CPU, skip `--use_cuda` or pass `--no-use_cuda` flag when running files `train.py` and `test.py`**


## GMM Training

Trained on GPU *NVIDIA GeForce RTX 3050* with *4GB RAM* and *CUDA Version: 12.2*.
Training took around 4hrs for 200K steps.

Train command:
```
python train.py --stage GMM --name train_gmm_200K --workers 4 --save_count 5000 --shuffle --use_cuda
```

Test command:
```
python test.py --stage GMM --name test_gmm_200K --workers 4 --datamode test --data_list test_pairs.txt --checkpoint checkpoints/train_gmm_200K/gmm_final.pth --use_cuda
```


## TOM Training

Trained on GPU *NVIDIA GeForce RTX 3050* with *4GB RAM* and *CUDA Version: 12.2*. Training took around 13hrs for 200K steps.

Generate wrap data for TOM train module:
```
python test.py --stage GMM --name generate_gmm_200K --workers 6 --datamode train --data_list train_pairs.txt --checkpoint checkpoints/train_gmm_200K/gmm_final.pth --use_cuda
```

Train command:
```
python train.py --stage TOM --name train_tom_200K --workers 6 --save_count 50000 --shuffle --use_cuda
```

**Make sure to copy test data generated by GMM module to`data/test/` folder before running test command on TOM.**

Test command:
```
python test.py --stage TOM --name test_tom_200K --workers 6 --datamode test --data_list test_pairs.txt --checkpoint checkpoints/train_tom_200K/tom_final.pth --use_cuda
```
