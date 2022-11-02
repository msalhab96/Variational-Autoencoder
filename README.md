# Variational-Autoencoder
A pytorch implementation of Variational Autoencoder (VAE) and  Conditional Variational Autoencoder (CVAE) on the MNIST dataset

An implementation of Conditional and non-condiational Variational Autoencoder (VAE), trained on MNIST dataset.



# Setup
1. Clon the repository 
```bash
git clone https://github.com/msalhab96/Variational-Autoencoder
```

2. create virtual enviroment 
```bash
python -m venv env
```

3. source the virtual enviroment
```bash
source env/bin/activate
```

4. install the requirements

```bash
pip install -r requirements.txt
```

# train the model
```bash
python train.py --batch_size 128 --conditional --latent_size 2
```


