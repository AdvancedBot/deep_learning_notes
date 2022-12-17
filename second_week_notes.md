# Second week

![image-20221215125553483](./second_week_notes.assets/image-20221215125553483.png)



## Basic of Neural network programming

当你在你的网络中组织计算的时候，经常使用 forward pause or forward propagation step以及backward pause or backward propagation step.

why the computations in learning an neural network can be organized in this forward propagation and a separate backward propagation.



### Logistic regression

logistic regression is an algorithm for binary classification.

![image-20221215160737296](./second_week_notes.assets/image-20221215160737296.png)

计算机对彩色图片采用的是RGB三通道。图片的大小是像素。

如果图片大小是64pixel*64pixel，三通道的矩阵为$3\cdot64\cdot64$, 64 by 64

![image-20221215161409775](./second_week_notes.assets/image-20221215161409775.png)

转换为12288*1的向量。

X.shape is 12288*1，X作为输入，得到label，y

![image-20221215165354316](./second_week_notes.assets/image-20221215165354316.png)



![image-20221215173620788](./second_week_notes.assets/image-20221215173620788.png)





## logistic regression

binary classification problems

![image-20221215180043066](./second_week_notes.assets/image-20221215180043066.png)

用线性函数来做二分类不好，因为可能大于1，或者小于0.

![image-20221215180200688](./second_week_notes.assets/image-20221215180200688.png)

![image-20221215181608468](./second_week_notes.assets/image-20221215181608468.png)

上图是sigmoid函数。

![image-20221215195704553](./second_week_notes.assets/image-20221215195704553.png)

sigmoid函数方程

Z如果是特别大的正数的话，$e^{-z}$会接近0，最后整个结果接近1。vice versa

![image-20221215200956224](./second_week_notes.assets/image-20221215200956224.png)



## Cost function

![image-20221215201952222](./second_week_notes.assets/image-20221215201952222.png)

superscript i (上标 i)，表示的是第i个。



Loss function:

![image-20221215202352113](./second_week_notes.assets/image-20221215202352113.png)

逻辑回归一般不用squared loss function，因为会得到很多局部最优解。

![image-20221215233950016](./second_week_notes.assets/image-20221215233950016.png)

![image-20221215234033372](./second_week_notes.assets/image-20221215234033372.png)





Cost function:

![image-20221215234256865](./second_week_notes.assets/image-20221215234256865.png)





## Gradient descent algorithm

![image-20221215234603955](./second_week_notes.assets/image-20221215234603955.png)



convex function：凸函数

![image-20221215234734193](./second_week_notes.assets/image-20221215234734193.png)

因为下面那个有很多不同的局部最优。



1. 随便赋一个初值。
2. 沿着最快下降方向，往下走。
3. 最后收敛于global optimum。



![image-20221216001041539](./second_week_notes.assets/image-20221216001041539.png)

编程的时候，一般用dw表示求导。

![image-20221216001331743](./second_week_notes.assets/image-20221216001331743.png)

只有一个变量用d，有两个及以上变量时，使用$\alpha$。叫做partial derivative symbol，偏导数符号。

![image-20221216002118751](./second_week_notes.assets/image-20221216002118751.png)





## Calculus and Derivatives

$f(x)=3x$，x=2, then f(2) = 6.

倒数就是这个三角形。

![image-20221216114438665](./second_week_notes.assets/image-20221216114438665.png)

编程的主要思想，就是让你求导数那个点加一个特别小的数，然后求斜率。

![image-20221216114759721](./second_week_notes.assets/image-20221216114759721.png)

![image-20221216135908678](./second_week_notes.assets/image-20221216135908678.png)



神经网络的计算都是按照forward path or forward propagation step in which we compute the output of the neural network.

首先计算出神经网络的输出，紧接着进行一个反向传输操作，后者我们用来计算出对应的梯度或者导数。





## Derivatives with a computation graph

复合函数求导，基本的求导式子。





## logistic regression gradient descent



![image-20221216172416330](./second_week_notes.assets/image-20221216172416330.png)

![image-20221216172232784](./second_week_notes.assets/image-20221216172232784.png)

minimize the loss function

计算一阶导数。log函数一般都是以e为底的。

![image-20221216172633791](./second_week_notes.assets/image-20221216172633791.png)



a = sigmoid(z) -> $\frac{da}{dz} =\frac{e^{-z}}{(1+e^{-z})^{2}} $将原式a = sigmoid(z)带入da/dz中得到 a(1-a)。

Sigmoid函数由下列公式定义

![image-20221216175210867](./second_week_notes.assets/image-20221216175210867.png)

其对x的导数可以用自身表示：

![image-20221216175220622](./second_week_notes.assets/image-20221216175220622.png)

![image-20221216223625071](./second_week_notes.assets/image-20221216223625071.png)

这个只是一个例子的梯度下降，我们要在每个训练集都要进行



## Gradient descent on m examples.

![image-20221216223822607](./second_week_notes.assets/image-20221216223822607.png)

求和然后求平均值。

就用for循环就行了。

![image-20221216224717061](./second_week_notes.assets/image-20221216224717061.png)



Two weaknesses: 

1. 使用两个for循环，第一个for loop遍历m个训练样本的小循环，第二个for循环时遍历所有特征的循环。在这个例子中我们只有两个特征，n=2, $n_{x}=2$, 如果有更多的话dw1,dw2....$dw_{n}$。使用for循环遍历这n个特征（W）。
2. 向量化操作加速运算，拜托for循环操作。





## Vectorization

其实就是使用python的numpy

```python
import numpy as np
a = np.array([1,2,3,4])
b = np.array([2,3,4,5])
```



Jupyter notebook使用CPU。

avoid using for loop



## More vectorization examples

When you are programming your neural networks, whenever possible avoid explicit for-loops.

![image-20221216233829838](./second_week_notes.assets/image-20221216233829838.png)

```python
import numpy as np
u = np.exp(v)
v**z
```





## Vectorizing logistic Regression

