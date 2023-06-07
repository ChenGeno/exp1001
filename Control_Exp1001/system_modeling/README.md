# System-modeling



## 利用该框架训练系统预测模型 

### 模型训练

``` python
CUDA_VISIBLE_DEVICES=3 python -m system_modeling.model_train dataset=southeast dataset.dataset_window=5 save_dir=tmp model=vrnn ct_time=True random_seed=0 train.batch_size=1024 train.epochs=1 train.eval_epochs=1 use_cuda=False
```
 使用3号GPU

> 参考```scripts/vrnn_multirun.sh```文件

 - 单模型训练：不加multirun参数

``` python
CUDA_VISIBLE_DEVICES=3 python -m system_modeling.model_train train.batch_size=32 dataset=winding model.D=1
```
#### 训练参数说明

运行脚本中添加 ```参数名=参数值```，即可设定参数，不同项之间用空格隔开。

| 参数              | 含义                       | 默认值   | 备注                                                         |
| ----------------- | -------------------------- | -------- | ------------------------------------------------------------ |
| --multirun        | 是否开启多轮实验           |          | 将多个给定参数用逗号隔开：model=a,b，将执行所有参数的笛卡尔积组合 |
| dataset           | 训练的数据集               | winding  | 需要在config/dataset下有其yaml配置文件，各数据集详细信息参见本章**数据集介绍** |
| model             | 模型                       | rssm     | 各模型详细信息参见本章**模型介绍**                           |
| random_seed       | 随机种子                   | 0        |                                                              |
| use_cuda          | 是否开启cuda               | True     | 代码没有实现多卡并行，只会使用0号GPU                         |
| save_dir          | 日志及模型存储路径关键词   | tmp      | 模型训练日志的存储路径详见**训练结果及日志存储**。调试代码时，将save_dir设置为tmp，可以避免最后统计实验结果时将调试模型加入对比中。 |
| ct_time           | 是否开启非均匀采样         | no       | 为True时，开启对训练数据和测试数据的下采样，并对保留时间点标记所属时间戳，开启后，external_input会自动增加一维。与其他几个参数配合使用。```sp```: 控制下采样比例，为1时代表不下采样。 <br/>```sp_even```：是否均匀采样，False代表不均匀。 <br/>```base_tp```：基础的时间间隔。 |
| train.epochs      | 训练轮次                   | 800      |                                                              |
| train.batch_size  | 批大小                     | 512      |                                                              |
| train.eval_epochs | 验证集的评估周期           | 10       | 每过train.eval_epochs轮，在数据集上进行一次评估              |
| train.min_epochs  | 最少训练轮次               | 200      | epoch超过train.min_epochs后，才可能会出发早停。              |
| train.eval_indice | 验证集评估指标             | rmse     | 可选集合：loss, rrse, rmse, pred_likelihood, multisample_rmse，根据该指标决定保存哪个训练轮次下的模型参数，以及决定是否早停。 |
| train.optim       | 优化器选择                 | adam     | adam或sgd，通过train.optim.lr=xxx 更改优化器参数             |
| train.schedule    | 学习率调整策略             | val_step | 可选：cosine、exp、none、step、val_step。推荐val_step        |
| test.plt_cnt      | 模型测试时的图片保存数量   | 20       | 每个图包含四部分，左上角：预测指标；右上角：重构图（只适用于VAE模型）；左下角：预测图；右下角：无意义 |
| test.test_dir     | 执行model_test时给定的路径 | None     | 对于等号'``` =```'、单引号'```'```' 等参数需要在前面加单斜杠 ‘ \’  进行转义。 |
| test.batch_size   | 测试时的batchsize          | 128      |                                                              |
| test.n_traj       | 预测的轨迹数               | 128      | 对于随机模型，支持多条预测轨迹随机采样                       |
| test.plt_n_traj   | 预测图中画出的轨迹数       | 4        | 画出的轨迹数不能超过test.n_traj                              |
| test.sp_change    | 测试时的sp                 | False    | 允许和训练时不同                                             |
| test.plt_single   | 是否画单独的预测图         | True     |                                                              |
|                   |                            |          |                                                              |
|                   |                            |          |                                                              |

### 模型测试

在训练完成后，框架会自动进行模型测试。如果需要再次测试，可以通过给定ckpt_path并运行```model_test```。如：

```bash
CUDA_VISIBLE_DEVICES=0 python -m system_modeling.model_test test.test_dir=\'/root/data/SE-VAE/ckpt/southeast/ct_True/ode_rssm/ode_rssm_ct_time\=True,model.k_size\=64,model.ode_solver\=rk4,model.ode_type\=orth,model.state_size\=32,random_seed\=0,sp\=0.5,train.batch_size\=1024/2022-04-30_06-20-37\'
```

> 对于 =、' 等参数需要在前面加单斜杠 ‘ \’  进行转义。

运行后将自动在当前目录下生成（或更新）```figs```文件夹以及```test.out```日志文件。



#### 训练结果及日志存储

训练模型与日志存储在```BASE_ROOT/ckpt/${dataset.type}/${save_dir}/${model.type}_${实验参数}/${now:%Y-%m-%d_%H-%M-%S}```中

训练完成后将自动绘制loss图，并在测试集上进行评估，目录包含的文件：测试之后会在目录下生成```figs```文件夹以及```test.out```日志文件

- figs: 测试集中部分数据的预测可视化结果，可修改test.plt_cnt。

- best.pth: 验证集score最高的模型参数。

- log.out: 训练过程日志。

- train_loss.png: 训练loss图。

- val_loss.png: 验证集loss和训练集loss对比图。

- test.out: 验证集中每条数据对的预测指标详情，以及测试集评估结果汇总。

- exp.yaml: 实验参数记录。

  

### 模型介绍

> 目前包含的离散时间域模型如下:

| 符号      | 模型                                |                    机理简化(预测过程)                    | 备注                                                       | 参考文献 |
| --------- | ----------------------------------- | :------------------------------------------------------: | :--------------------------------------------------------- | :------- |
| ~~vaecl~~ | ~~VAE combinational linears~~       |                                                          | ~~动态系统为自适应线性组合，线性解器~~，效果一般已经被废弃 |          |
| vrnn      | variational RNN                     | $h_{i+1}=f(h_i,x_i,y_i,z_i)\\z_i=p(h_i)\\y_i=g(h_i,z_i)$ |                                                            | [1]      |
| srnn      | stochastic recurrent neural network |                                                          |                                                            | [2]      |
| seq2seq   | attention seq2seq                   |     $h_N=f(x_{1:N})\\y_{N+1:N+M}=g(h_N,x_{N+1:N+M})$     |                                                            | [3]      |
| deepar    | Deep autoregressive recurrent       |                                                          |                                                            | [4]      |
| informer  |                                     |   $h_N=ATT(x_{1:N})\\y_{N+1:N+M}=ATT(h_N,x_{N+1:N+M})$   | 基于Transformer的序列预测模型                              | [5]      |
| nn        | neural network                      |                                                          | 普通神经网络，$s_t,a_t->s_{t+1}$                           |          |
| rnn       | Recurrent  neural network           |                                                          | 普通的RNN                                                  |          |
| rssm      | recurrent state space model         |                                                          |                                                            | [6]      |
| storn     | Stochastic RNN                      |                                                          |                                                            | [7,9]    |
| vaernn    | variational  Auto-encoder RNN       |                                                          |                                                            | [8.9]    |
|           |                                     |                                                          |                                                            |          |

> 连续时间模型如下：

| 符号       |                            模型                            |                      机理简化(预测过程)                      |   备注   | 参考文献 |
| :--------- | :--------------------------------------------------------: | :----------------------------------------------------------: | :------: | :------: |
| time_aware |                         Time-Aware                         |                                                              |          |   [10]   |
| ode_rnn    |                          ODE-RNN                           |                                                              |          |   [11]   |
| latent_sde |                         Latent sde                         |                                                              |          | [12,13]  |
| ode_rssm   | Ordinary Differential Equation Recurrent State Space Model | $\tilde{\boldsymbol{h}}_{t_{i-1}}=\operatorname{GRU}\left(\left[\boldsymbol{u}_{t_{i-1}}, \boldsymbol{z}_{t_{i-1}}\right], \boldsymbol{h}_{t_{i-1}}\right)\\\boldsymbol{h}_{t_i}=\operatorname{ODE}\left( \tilde{\boldsymbol{h}}_{t_{i-1}}, \mathrm dt\right)\\\boldsymbol{z}_{t_i}\sim p\left(\boldsymbol{z}_{t_i} \mid \boldsymbol{h}_{t_i}\right)$ | 本组工作 |   [14]   |

### 数据集介绍

|  数据集符号   |                             描述                             | 输入-输出大小 | 来源 |
| :-----------: | :----------------------------------------------------------: | ------------- | :--: |
|     cstr      |                    **连续搅拌釜式反应器**                    | 1-2           | [10] |
|    winding    |                      **工业绕组数据集**                      | 5-3           | [10] |
|   southeast   |                  东南矿体泥层压力变化数据集                  | 4-1           | [14] |
|   actuator    |                                                              | 1-1           | [15] |
|   ballbeam    |                                                              | 1-1           | [15] |
|     drive     |                                                              | 1-1           | [15] |
|     dryer     |                                                              | 1-1           | [15] |
|  gas_furnace  |                                                              | 1-1           | [15] |
|      ib       |                 工业强化学习控制的benchmark                  | 3-6           | [16] |
|      nl       |                                                              | 1-1           | [9]  |
|    sarcos     |                                                              | 7-7           | [15] |
|  southeast32  |         东南矿体浓密机数据集(出料浓度和泥层作为输出)         | 3-2           | [14] |
| thickener_sim |                                                              |               | [17] |
|     west      | 西矿体泥层压力数据集，分成了多个文件，每个文件对应连续生产过程 | 8-1           |      |
|   west_con    |          西矿体浓密机数据集（出料浓度和压力为输出）          | 5-2           | [18] |
|               |                                                              |               |      |




### jupyter脚本
- ```jupyter/generate_data_frame.ipynb```：生成所有数据集的所有模型的对比结果(不考虑```ckpt/${dataset.type}/tmp/``` 目录下的模型)
- ```jupyter/delete_error_ckpt.ipynb```: 自动检测并删除训练失败的日志目录，顺带删除```ckpt/${dataset.type}/tmp/``` 下的文件
- ```jupyter/find_error_ckpt.ipynb```: 自动检测训练失败的日志目录，不删除。



## 其他

### hydra  

框架运行的所有参数配置通过facebook开发的hydra实现。

参考资料: https://hydra.cc/



### 该框架与有模型强化学习的衔接

创建SystemModel的初衷为了做有模型强化学习的建模部分，与Control_Exp1001联动，构建system modeling包的对外访问接口：

由于system modeling 中假设状态的转移过程为 $s_t, a_t \rightarrow s_{t+1}$，而非 $s_{t-1}, a_t \rightarrow s_t$ ， **明确这一点对于理解为什么要构建额外封装非常重要 !!!**

```BASE_ROOT/model_render.py``` 文件提供了SystemModel作为一个独立的包的对方访问结构，详情请阅读该文件的注释代码。具体使用案例可参考仓库[Control_Exp1001](https://codeup.aliyun.com/618ce291f1ae9b61971dbe66/y18810919727/Control_Exp1001)。

> 之所以```system modeling```中```model_train.py```和```model_test.py```的启动方式要替换成```python -m``` 的形式，也是为了解决对外提供访问接口问题。所有的文件引用都变成了``` from . import yyy``` 和```from .xxx import yyy```的包内引用，避免了sys.path不确定时import不到文件的情况。



### 如何添加数据集

#### 新增的数据集应满足以下条件：

1. 是一个低维的多输入-多输出系统数据集，尚不支持高维观测（比如图像观测）
2. 给定数据应满足转换关系，$(s_t,a_t)\rightarrow s_{t+1}$，避免控制量和输出量错位

#### 操作过程

1. 在```config/dataset/```下添加新数据集的配置文件

   1. 每项输出要给定target_names
   2. ```history_length+forward_length``` 代表训练切片对的长度

2. 在```data/```下创建文件夹，放入数据集源文件。

   > 如果文件特别大，建议在代码里添加文件自检+远程下载，然后在.gitignore里添加忽略，不要放到托管git仓库托管。

3. 在```dataset.py```中添加构建数据集的class，并在```prepare_training_dataset```中添加训练集、验证集的载入代码

4. 在```model_test.py```的```main_test```中添加载入测试集的代码。



### 如何添加模型

#### 新增的模型应满足以下条件：

1. 建模$(s_t,a_t)\rightarrow s_{t+1}$，避免控制量和输出量错位。

#### 操作过程

1. 在```config/model/```下添加新模型的默认配置文件，里面的参数可以任意指定
2. 在```model```目录下添加模型代码，如果是离散时间模型直接加在```model```下，如果是连续时间模型，添加在```model/ct_model```下。如果代码比较复杂，可以直接新加一个package。主模型class需要继承```model.BaseModel```，要实现的方法参见本节——**模型至少需要实现的方法**
3. 在```model/__init__.py```下添加新加model的import，注意使用相对引用。
4. 在```model/generate_model.py```下添加对应的模型实例化过程。

#### 模型至少需要实现的方法

这一部分在```model/base_model.py```中也做出了说明，注意其中代码中存在```NotImplementedError```的地方。

##### 初始化函数

```__init__()```：初始化函数，根据需要随意构造，初始化环节不需要考虑模型在cpu还是gpu上。

##### 预测函数

```
def _forward_prediction(self, external_input_seq, n_traj, memory_state=None)
```

**输入参数**

- ```external_input_seq``` (```torch.FloatTensor```) : 外部控制输入序列，shape为 ```(L, batch_size, dim_in)```
- ```n_traj```(```int```): 预测采样轨迹

**返回值**

**方法含义**

模型根据由历史序列编码得出的```memory_state```，以及给定```external_input_seq```假定其大小为```(len, batch_size, dim_in)```，想要预测的系统输出为```pred_observations_seq```，其中```pred_observations_seq```和```external_input_seq```等长。

函数返回值包括两部分：

- 第一部分应该是一个字典（outputs）：其中至少包含predicted_seq_sample, predicted_dist, predicted_seq三个key，分别对应

  - ```predicted_seq_sample (len, batch_size, n_traj, dim_out)```： 预测分布的采样
  - ```predicted_dist```：```MultivariateNormal```的实例，均值loc的大小为```(len, batch_size, n_traj, dim_out)```，协方差矩阵的大小为```(len, batch_size, n_traj, dim_out, dim_out)```
  - ```predicted_seq```(len, batch_size, dim_out)```： 单个预测序列，一般情况下可以返回predicted_dist的均值。

  

- 第二部分是```memory_state```，代表模型输入```external_input_seq```之后的对于old_memory_state的更新。其设计的初衷是为了满足如下设定：

  ```
  m0: 初始的memory state
  
  Case 1: 分段预测
  ya, ma = model.forward_prediction(xa, m0)
  yb, mb = model.forward_prediction(xa, m0)
  
  Case 2: 整体预测
  yab, mab = model.forward_prediction(cat([xa, xb], m0)
  
  需要满足：
  1.  yab == cat([ya,yb])
  2.  mab == mb
  ```

  memory_state内部参数的设计需要满足上述设定，且满足```_forward_posterior```中对于memory_state的要求。

##### 后验推理函数

```
def _forward_posterior(self, external_input_seq, observations_seq, memory_state=None):
```

这一函数设计的初衷只是为了满足时序变分自动编码器的重构需求，进行模型训练。然而对于其他大部分确定性模型（如RNN等）是不需要的。 是但是为了保证接口统一，所有模型都需要实现该函数。

##### 解码重构函数

```
def decode_observation(self, outputs, mode='sample'):
```



这一函数是与```_forward_posterior```配套使用的，方法```_forward_posterior```的返回值的第一部分——```outputs```(一个字典)是```decode_observation```的输入，字典结构可以根据模型需要随意设计，只要满足以下设定即可：

```
output, memory_state = self.forward_posterior(X, Y)
Y_reconstruct = self.decode_observation(output, mode) # mode可以为dist 或 sample

```

```Y_reconstruct```是```Y```的重构分布或者从分布中的采样，由mode决定

> 对于非VAE、非AE模型，其本质上是没有重构这一概念的，只要函数返回与Y的shape相同的序列或分布即可。返回分布中的协方差矩阵可以直接设置为零矩阵。

### 下一步值得复现并加入模型库的模型

1. Probabilistic recurrent state-space models$^{[15]}$: 高斯过程模型和状态空间模型的结合，现有框架里还未尝试引入高斯过程模型
2. SNODE$^{[19]}$:  带有gradient matching的连续时间模型，理论上讲训练效率会远高于ODE-RSSM和ODE-RNN
3. Latent ODE$^{[11]}$: 无法适用于在线问题，但是可以类似于seq2seq/informer的处理方式，把历史序列都存起来，或者模型只应用于离线预测
4. VSDN$^{[12]}$:  目前实现的Latent SDE带了一点VSDN的影子，但是效果相比VSDN肯定还是差强人意。
5. Latent CTFP$^{[12]}$: 将连续归一化流+ODE+随机过程 杂糅在一起的时间序列生成模型，论文给出的结果是比Latent ODE好不少。
6. AJ-ODE-Net：本组发表在TII上的工作《Autonomous-Jump-ODENet: Identifying Continuous-Time Jump

   Systems for Cooling-System Prediction》，其本质也是序列编码+预测的形式，可以加入到模型库。(AJ-ODE-Net地址：https://github.com/y18810919727/cooling)

### 本项目需要完善的地方

本项目代码还有一些地方需要完善

1. 预测序列长度完全是由args.dataset指定的，测试阶段不支持变长度的序列预测，无法实现AJ-ODE-Net代码中的长序列仿真效果
2. 如果AJ-ODE-Net要加入到仓库，需要在仓库中考虑多阶段系统辨识问题，可以类似于配置文件中的```ct_time```，再加上一个多阶段数据集的标识，将阶段变量作为```external_input```的最后一位。
3. 目前代码只能从头开始训练，不支持load现有模型进行fine-tune。

## 参考文献

[1] Chung, J., Kastner, K., Dinh, L., Goel, K., Courville, A., & Bengio, Y. (2015). A recurrent latent variable model for sequential data. *Advances in Neural Information Processing Systems*, *2015*-*Janua*, 2980–2988.

[2] Fraccaro, M., Sønderby, S. K., Paquet, U., & Winther, O. (2016). Sequential neural models with stochastic layers. *Advances in Neural Information Processing Systems*, 2207–2215.

[3]  https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

[4] Valentin Flunkert, David Salinas, and Jan Gasthaus. DeepAR: Probabilistic forecasting with autoregressive recurrent networks. CoRR, abs/1704.04110, 2017. URL http://arxiv. org/abs/1704.04110.

[5] Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., & Zhang, W. (2020). Informer: Beyond efficient transformer for long sequence time-series forecasting. *ArXiv*.

[6] Hafner, D., Lillicrap, T., Fischer, I., Villegas, R., Ha, D., Lee, H., & Davidson, J. (2019). Learning latent dynamics for planning from pixels. *36th International Conference on Machine Learning, ICML 2019*, *2019*-*June*, 4528–4547.

[7] Bayer, J. and Osendorfer, C. (2015). Learning Stochastic Recurrent Recurrent Networks. arXiv:1411.7610.

[8] Fraccaro, M. (2018). Deep Latent Variable Models for Sequential Data. Ph.D. thesis, DTU Compute. 

[9] Gedon, D., & Wahlstr, N. (n.d.).  Deep State Space Models for Nonlinear System Identification.

[10] Demeester, T. (2019). System Identification with Time-Aware Neural Sequence Models. Retrieved from http://arxiv.org/abs/1911.09431

[11] Rubanova, Y., Chen, R. T. Q., & Duvenaud, D. (2019). Latent ODEs for irregularly-sampled time series. *ArXiv*.

[12] Liu Y, Xing Y, Yang X, et al. Continuous-Time Stochastic Differential Networks for Irregular Time Series Modeling[C]//International Conference on Neural Information Processing. Springer, Cham, 2021: 343-351.

[13] Li, X., Leonard Wong, T. K., Chen, R. T. Q., & Duvenaud, D. (2020). Scalable gradients for stochastic differential equations. *ArXiv*, *108*.

[14] **Yuan Zhaolin** , et al. ODE-RSSM: Learning Stochastic Recurrent State Space Model from Irregularly Sampled Data. In Proceedings of the 37th AAAI Conference on Artificial Intelligence, 2023

[15] Doerr A, Daniel C, Schiegg M, et al. Probabilistic recurrent state-space models[C]//International Conference on Machine Learning. PMLR, 2018: 1280-1289.

[16] Hein D, Depeweg S, Tokic M, et al. A benchmark environment motivated by industrial control problems[C]//2017 IEEE Symposium Series on Computational Intelligence (SSCI). IEEE, 2017: 1-8.

[17]. **袁兆麟**, 何润姿, 姚超, 等. 基于强化学习的浓密机底流浓度在线控制算法 [J]. 自动化学报, 2021, 47(7): 1558-1571.

[18]. **Yuan Z**, Li X, Wu D, et al. Continuous-time prediction of industrial paste thickener system with differential ODE-net[J]. IEEE/CAA Journal of Automatica Sinica,2022, 9(4): 686-698.

[19]. QUAGLINO A, GALLIERI M, MASCI J, et al. SNODE: Spectral Discretizationof Neural ODEs for System Identification[C/OL] // 8th International Conference on Learning Representations, ICLR 2020,

[20]. Deng R, Chang B, Brubaker M A, et al. Modeling continuous stochastic processes with dynamic normalizing flows[J]. Advances in Neural Information Processing Systems, 2020, 33: 7805-7815.

## 

