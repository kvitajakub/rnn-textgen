Metacentrum
===========

* maji Intel CDK - BLAS

Login
-----
* login: `xkvita01`
* heslo jako na Fitu
* [seznam uzlu zde](https://wiki.metacentrum.cz/wiki/%C4%8Celn%C3%AD_uzel)
* `ssh xkvita01@skirit.metacentrum.cz`


Instalace Torche
----------------
Naloadovani potrebnych modulu pro instalaci a nastaveni cesty u MKL, ktery se nastavuje jinak.
```
module add cmake-3.2.3
module add intelcdk-15
module add openmpi-intel
export CMAKE_LIBRARY_PATH=$CMAKE_LIBRARY_PATH:$MKLROOT/lib/intel64
```
LuaJIT jako interpret
```
git clone https://github.com/torch/distro.git /storage/brno2/home/xkvita01/torch --recursive
cd /storage/brno2/home/xkvita01/torch
./install.sh
. /storage/brno2/home/xkvita01/torch/install/bin/torch-activate
```
Lua 5.2 jako interpret
```
git clone https://github.com/torch/distro.git /storage/brno7-cerit/home/xkvita01/torch --recursive
cd /storage/brno7-cerit/home/xkvita01/torch
TORCH_LUA_VERSION=LUA52 ./install.sh
. /storage/brno7-cerit/home/xkvita01/torch/install/bin/torch-activate
```
Potreba updatovat nn balik, kvuli nejakym problemum s Cudou.
```
luarocks remove nn
luarocks install nn
```
Doinstalovani baliku pro CUDA a rekurentni site...
```
module add cuda-6.5
luarocks install cutorch
luarocks install cunn
luarocks install cunnx
luarocks install rnn
luarocks install tds
luarocks install profi
```

Updatovani Lua baliku, nechce se mi hledat jak to delat
```
luarocks remove rnn
luarocks remove cunnx
luarocks remove cunn
luarocks remove cutorch
luarocks remove optim
luarocks remove dpnn
luarocks remove nngraph
luarocks remove nn

luarocks install nn
luarocks install optim
luarocks install cutorch
luarocks install cunn
luarocks install cunnx
luarocks install rnn
luarocks install nngraph
luarocks install dpnn
```

```
luarocks remove cunn --force
luarocks remove cutorch --force
luarocks install cutorch
luarocks install cunn
```

```
luarocks remove image --force
luarocks install image
```

Instalace Tmuxu
---------------
```
wget https://raw.githubusercontent.com/jealie/install_tmux/master/install_tmux.sh
chmod +x install_tmux.sh
./install_tmux.sh
export PATH=$PATH:/storage/brno2/home/xkvita01/local/bin
```


Naloadovani Torche a Tmuxu pri novem prihlaseni
---------------------------------------
* Nejaka prace s `.bashrc` v Metacentru nefunguje. Nasledujici prikazy je potreba zadat pokazde. Napr. v spoustecim skriptu.

### Tmux
```
export PATH=$PATH:/storage/brno2/home/xkvita01/local/bin
```
### Torch
```
cd /storage/brno2/home/xkvita01/
module add cmake-3.2.3
module add intelcdk-15
module add openmpi-intel
module add cuda-6.5
```
Vybereme jaky interpret chceme

Lua 5.2
```
. /storage/brno2/home/xkvita01/torch/install/bin/torch-activate
```
LuaJIT
```
. /storage/brno7-cerit/home/xkvita01/torch/install/bin/torch-activate
```
### Vlakna
```
export OMP_NUM_THREADS=1
```

### Dohromady
```
export PATH=$PATH:/storage/brno2/home/xkvita01/local/bin
cd /storage/brno2/home/xkvita01/
module add cmake-3.2.3
module add intelcdk-15
module add openmpi-intel
module add cuda-6.5
export OMP_NUM_THREADS=1
. /storage/brno2/home/xkvita01/torch/install/bin/torch-activate
```

Kopirovani dat
--------------
```
tar -cf source.tar network.lua readFile.lua sampling.lua
scp source.tar xkvita01@skirit.metacentrum.cz:~/
tar -cf data.tar text/
scp data.tar xkvita01@skirit.metacentrum.cz:~/
```
```
scp network.lua xkvita01@skirit.metacentrum.cz:~/char-predict/
```

Vzdalene
```
tar -xf source.tar
tar -xf data.tar
```


Vytvareni uloh pres qsub
------------------------
Pres vytvarec [online](http://metavo.metacentrum.cz/pbsmon2/person). Pridat prepinac `-I` pro interaktivni rezim nebo pripojit spousteci skript.
Napr. 1 cpu na 24 hodin.
```
qsub -I
```
Uzel s GPU (max. 1 den)
```
qsub -q gpu -l mem=6gb -l nodes=1:ppn=2:gpu=1:^cl_konos -I
qsub -q gpu -l mem=6gb -l nodes=1:ppn=2:gpu=2:cl_doom -I
qsub -q gpu -l mem=6gb -l nodes=1:ppn=2:gpu=2:cl_gram -I
qsub -q gpu -l mem=6gb -l nodes=1:ppn=2:gpu=1:cl_zubat -I
```
Uzel s GPU (na dlouho - tyden (cluster doom))
```
qsub -l walltime=7d -q gpu_long -l mem=6gb -l nodes=1:ppn=2:gpu=2:cl_doom -I
```

Kontrola
--------
Zkontrolovat to pres htop
```
htop
```
Pozadi/popredi
```
# zastavit pres CTRL+Z
bg
```
```
fg
```
Vytizeni GPU
```
nvidia-smi
```
Ktere GPU je moje
```
/usr/sbin/list_cache arien gpu_allocation | grep $PBS_JOBID
```

COCO dataset
------------
Dataset je ulozeny v
```
cd /storage/brno7-cerit/home/xkvita01/COCO
```
obrazky jsou v podadresarich `train2014` a `val2014`, jsony s popiskama jsou tamtez.

CNN natrenovana sit
```
scp nin.torch  xkvita01@storage-brno7-cerit.metacentrum.cz:~/CNN
```
### Konvolucni site maji cestu:
```
cd /storage/brno7-cerit/home/xkvita01/CNN/
```
### Rekurentni site maji cestu:
```
cd /storage/brno7-cerit/home/xkvita01/RNN/
```


th training.lua -pretrainedCNN /storage/brno7-cerit/home/xkvita01/CNN/VGG_ILSVRC_16_layers.torch  -pretrainedRNN /storage/brno7-cerit/home/xkvita01/RNN/1.0000__3x200.torch -batchSize 15 -printError 10 -sample 100 -saveModel 2000 -modelName VGG_3x200.torch
