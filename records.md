##### 机器学习的核心思想
* 使用现有的数据，训练出一个模型，然后再用这样的一个模型去拟合其他的数据，给未知的数据做出一个预测。

1. 统计机器学习
2. BP神经网络
3. 深度学习

##### 机器学习的种类
1. 监督学习
    * 学习一个模型，使用模型能够对任意给定的输入做出相应的预测；学习的数据形式是（X,Y）组合。
2. 无监督学习（介于二者之间的半监督学习）
    * 学习一个模型，使用的数据是没有被标记过的，自己默默地在学习隐含的特征，寻找模型与规律。输入数据形式只有X.例如聚类。
3. 强化学习
    * 在没有指示的情况下，算法自己评估预测结果的好坏。从而使得计算机在没有学习过的问题上，依然具有很好的泛化能力。

##### 损失函数
* 在数学上找到衡量预测结果与实际结果之间偏差的一个函数。通过反复迭代地训练模型，学习特征，是偏差极小化。机器学习是一个求解最优化问题的过程。

##### 训练模型应该避免的两种情况
* 过拟合：模型训练过度，假设过于严格。
* 欠拟合：模型有待继续训练，拟合能力不强。

##### MLlib是Spark的机器学习（ML）库,它提供了以下工具：
1. ML算法：常见的学习算法，如分类，回归，聚类和协同过滤
2. 特征化：特征提取，转换，降维和选择
3. 管道：用于构建，评估和调整ML管道的工具
4. 持久性：保存和加载算法，模型和管道
5. 实用程序：线性代数，统计，数据处理等。

##### MLlib的数据格式
* 本地向量。本地向量是存储在本地节点上的，其基本数据类型是Vector.其有两个子集，分别是密集的与稀疏的，我们一般使用Vetors工厂来生成，例如：
    1. Vectors.dense(1.0,2.0,3.0)
    2. Vectors.sparse(3,(0,1),(1,2),(2,3))
##### 标签数据
* 监督学习是（x,y）数据形式。X是特征向量，y是标签。

##### 本地矩阵
* 与向量相似，本地矩阵类型为Matrix,分为稠密向量和稀疏向量两种类型。同样使用工厂方法Matrix来生成。但是要注意，Mlib的矩阵是按列存储的。例如下面创建的一个3*3的单位矩阵：
    * Matrices.dense(3,3,Array(1,0,0,0,1,0,0,0,1)) 稀疏
    * Matrices.sparse(3,3,Array(0,1,2,3),Array(0,1,2),Array(1,1,1)) 稠密

##### 分布式矩阵
* 分布式矩阵为把一个矩阵数据分布式存储到多个RDD中。将分布式矩阵进行数据转换需要全局的shuffle函数，最基本的分布式矩阵是RowMatrix.

##### 分布式数据集
1.RDD是spark中结构最简单，也是最常用的一类数据集形式。可以理解为把输入数据进行简单的封装之后形成的对内存数据的抽象。按行存储，只有一列，N行乘以1列。
2.与RDD分行存储，没有列的概念不同，DataSet引入了列的概念，这一点类似于一个CSV文件结构。类似于一个简单的二维表。
3.DataFrame结构与Dataset是类似的，都引入了列的概念。与Dataset不同的是，DataFrame中的每一行被再次封装为Row的对象。需要通过该对象的方法来获取到具体的值。

##### MLlib与ml的区别
1.MLlib采用RDD形式的数据结构，而ml使用DataFrame的结构
2.spark官方希望用ml逐步替代MLlib

##### 矩阵与向量介绍
1. 向量既有方向又有大小（模长）
2. 矩阵中的每一行可以看做是一个行向量，每一列就是一个列向量。
3. 向量也可以进行加减乘除等运算，与矩阵操作相似。
4. 向量还有范数的概念。

 ##### MLlib创建向量是按列排列的。
 ```
 1. val m1 = org.apache.spark.mllib.linalg.Matrices.dense(2,3,Array(1,4,2,5,3,6)) 按列创建分布式
 2. val m2 = breeze.linalg.DenseMatrix(Array(1,2,3),Array(4,5,6)) 按行创建 单机
 ```

##### spark基础统计知识
* 常用的统计学知识
    * 描述性统计：平均数、方差、众数、中位数......
    * 相关性度量：Spark提供了皮尔逊和斯皮尔曼相关系数，反映变量相关关系的密切程度。
    * 假设检验：根据一定的假设条件，由样本推断总体的一种统计学方法。spark提供了皮尔森卡方检测。

 ```
 import org.apache.spark.mllib.{stat, linalg}
 
 val txt = sc.textFile("data.txt")
 val data = txt.flatmap(_.split(",").map(value => linalg.Vectors.dense(_.toDouble())))
 val result = stat.Statistics.colStats(data)
 result.max
 ```
 
 ##### 相关性度量
 * 是一种研究变量之间线性相关程度的量
 * 主要介绍皮尔逊相关系数
 
 ##### 统计假设检验
* 根据一定的假设条件，由样本推断总体的一种统计学方法。基本思路是先提出假设，使用统计学方法进行计算，根据计算结果判断是否拒绝假设。
* 检验方法：卡方检验、T检验等。
* spark实现的是皮尔森卡方检验，它可以实现适配度检测和独立性检测。
* 皮尔森卡方检验是最常用的卡方检验，可以分为适配度检验和独立性检验。
    * 适配度检验：验证观察值的次数分配与理论值是否相等
    * 独立性检验：两个变量抽到的观察值是否相互独立

#### 回归分析
* 回归与分类类似，只不过回归的预测结果是连续的，而分类的预测结果是离散的。
* 这样就使得很多回归与分类的模型可以经过改动而通用
* 正因如此，对于回归和分类中基本原理相同或类似的模型，我们不在具体讲解

#### 线性回归
 * 在线性分析中，自变量和因变量之间满足或基本满足线性关系，可以使用线性模型进行拟合。
 * 如回归分析中，只有一个自变量的即为一元线性回归，其自变量与因变量之间的关系可以用一条直线近似的表示。
 * 同理，对于多变量的回归称为多元线性回归，其可以用一个平面或超平面来表示。
 
#### 使用线性回归的前提条件
* 自变量和因变量之间具有线性趋势，我们在前面介绍过相关系数。
* 独立性：自变量之间取值相互独立，不存在关联。

#### 回顾机器学习模型
* 对于统计学习来讲，机器学习模型就是一个函数表达式，其训练过程就是在不断更新这个表达式的参数，以便使这个函数能够对未知数据产生最好的预测效果。

#### 何为好的预测效果
+ 代价函数或者叫损失函数

#### 线性回归
* 线性回归是最简单的数学模型之一
* 线性回归的步骤是先用已有的数据，探索自变量X和因变量Y之间存在的关系，这个关系就是线性回归模型中的参数。有了它，就可以预测未知数据。
* 机器学习的模型的基本训练过程也是是如此，属于监督学习

#### 最小二乘法
* 最小二乘法又称最小平方法，通过最小化残差平方和来找到最佳函数匹配。
* 也就是说，最小二乘法以残差的平方和作为损失函数，用于衡量模型的好坏。
* 利用最小二乘法可以实现对曲线的拟合。

#### 随机梯度下降
* 是一种机器学习中常用的优化方法
* 它是通过不断迭代更新的手段，来寻找某一个函数的全局最优解的方法。
* 与最小二乘法类似，都是优化算法，随机梯度下降特别适合变量众多，受控系统复杂的模型，尤其是在深度学习中具有十分重要的作用。 
* 随机梯度下降的随机体现在进行梯度计算的样本是随机抽取的n个，与直接采用全部样本相比，这样计算量更少。

#### 线性回归代码实例
```
val conf = new SparkConf().setAppName("MLlib").setMaster("local[2]")
val sc = new SparkContext(conf)
val spark = SparkSession.builder().config(conf).getOrCreate()

val file = spark.read.format("csv").option("sep", ";").option("header", "true").load("/home/zero/IdeaProjects/linear/house.csv")

import spark.implicits._

val random = new util.Random()
val data = file.select("square", "price")
  .map(row => (row.getString(0).toDouble, row.getString(1).toDouble, random.nextDouble()))
  .toDF("square","price", "random").sort("random")
data.show()  // 强制类型转换

val assembler = new VectorAssembler().setInputCols(Array("square")).setOutputCol("features")

val dataset = assembler.transform(data)
var Array(train, test) = dataset.randomSplit(Array(0.8,0.2), 1234L)

train.show()
println(test.count())

val regression = new LinearRegression().setMaxIter(100).setRegParam(0.3).setElasticNetParam(0.8)
/*
 * fit 做训练
 * transform 做预测
 */
val model = regression.setLabelCol("price").setFeaturesCol("features").fit(train)
model.transform(test).show()
```

#### 逻辑回归算法概述
* 线性VS非线性
    * 线性简言之就是两个变量之间存在一次方函数关系
    * 自然界中变量间更多的关系是非线性的，绝对的线性关系相对较少。
    * 因此，在选择数学模型进行拟合的时候，很多情况使用非线性函数构造的模型可能比线性函数模型更好。
* 逻辑回归是logistic回归，是一种广义的线性回归，但是与线性回归模型不同的是，其引入了非线性函数。因此，逻辑回归可以用于非线性关系的回归拟合，这一点是线性回归所不具备的。

#### 模型是不是训练得越多越好吗？
* 实际上并不是这样的
* 过拟合、欠拟合和刚刚好
    * 对于欠拟合状态，只需要加大训练轮次，增加特征量，使用非线性模型等即可。
    * 常用的减少过拟合的方法有交叉验证法，正则化方法等。

#### 交叉验证法
* 所谓的交叉验证法，就是在训练过程中，将训练数据集拆分为训练集和验证集两个部分，训练集专用训练模型，验证集只为检验模型预测能力，当二者同时达到最优时，既是模型最优的时候。

#### 正则化原理
* 对模型的复杂程度进行量化，越复杂的模型，就越对其进行“惩罚”，以便使模型更加“中庸”。通过动态调节惩罚程度，来防止模型过于复杂。

#### 逻辑回归代码
``` 
val lr = new LogisticRegression()
      .setLabelCol("price")
      .setFeaturesCol("features")
      .setRegParam(0.3)
      .setElasticNetParam(0.8).setMaxIter(10)
val model = lr.fit(train)
model.transform(test).show()
val s = model.summary.totalIterations
println(s"iter: $s")
```

#### 保序回归
* 保序回归是用于拟合非递减数据（非递增一样）的一种回归分析，同时，保序回归能够使得拟合之后的误差最小化。
* 保序回归用于拟合非递减数据，不需要事先判断线性与否，只需要数据总体的趋势是非递减的即可。例如研究某种药物的使用剂量与药效之间的关系。
* spark实现求解该模型的算法是pool adjacent violators算法（PAVA）
```
val isotonic = new IsotonicRegression()
  .setLabelCol("price")
  .setFeaturesCol("features")
  
val model = isotonic.fit(train)
model.transform(test).show()
```

#### 朴素贝叶斯算法
* 朴素贝叶斯算法的基本假设是条件独立性，这是一个较强的前提条件，因而它易于实现，但是分类性能可能不会很高。
* 朴素贝叶斯算法要求输入变量是条件独立的，但是如果他们之间存在概率依存关系，就会超出该算法的范畴，属于贝叶斯网络。
```
 val conf = new SparkConf().setMaster("local").setAppName("iris")
val spark = SparkSession.builder().config(conf).getOrCreate()
spark.sparkContext.setLogLevel("WARN") ///日志级别

val file = spark.read.format("csv").load("/home/zero/IdeaProjects/linear/iris/iris.data")
//file.show()

import spark.implicits._
val random = new Random()
val data = file.map(row =>{
  val label =  row.getString(4) match {
    case "Iris-setosa" => 0
    case "Iris-versicolor" => 1
    case "Iris-virginica" => 2
  }

  (row.getString(0).toDouble,
    row.getString(1).toDouble,
    row.getString(2).toDouble,
    row.getString(3).toDouble,
    label,
    random.nextDouble())
}).toDF("_c0","_c1","_c2","_c3","label","rand").sort("rand")//.where("label = 1 or label = 0")

val assembler = new VectorAssembler().setInputCols(Array("_c0","_c1","_c2","_c3")).setOutputCol("features")

val dataset = assembler.transform(data)
val Array(train,test) = dataset.randomSplit(Array(0.8,0.2))
val bayes = new NaiveBayes().setFeaturesCol("features").setLabelCol("label")
val model = bayes.fit(train) //训练数据集进行训练
model.transform(test).show() //测试数据集进行测试，看看效果如何
```
#### 支持向量机
* 支持向量机(SVM)用来分类的算法，当然，在这基础上进行改进，也可以进行回归分析（SVR）
* SVM是最优秀的分类算法之一，即使是在如今深度学习盛行的时候，仍然具有很广泛的应用。
* SVM被设计成一种二分类的算法，当然，也有人提出了使用SVM进行多分类的方法，但是SVM依然主要被用在二分类中。
* SVM的主要思想是寻找能够将数据进行分类的平面或者超平面，在平面上的则是A类，在平面下的则是B类，因此，SVM是一种二分类算法。
* 因此，这个阀值更贴切地说应该称为边界，而这个边界恰恰是通过向量来表示的，故而边界我们就称为支持向量。
* 核函数
    * SVM虽然只能进行线性分类，但是，可以通过引入核函数，将非线性的数据，转化为另外一个空间中的线性可分数据，这叫做支持向量机的核技巧。（希尔博客空间）
* 基于硬间隔最大化的线性可分支持向量机（绝对分割）
* 基于软间隔最大化的线性支持向量机（有部分遗漏）
```
// 原始数据只有处理成两类，否则报错! 
代码.sort("rand").where("label = 1 or label = 0")
val svm = new LinearSVC().setLabelCol("label").setFeaturesCol("features").setMaxIter(10).setRegParam(0.1)
val model = svm.fit(train)
model.transform(test).show()
```

#### 决策树
* 决策树因其进行决策判断的结构与数据结构中的树相同，故而得名
* 决策树可以实现分类，也可以实现回归，一般用作分类的比较多。例如if-then就是一种简单的决策树。
* 决策树的解法有很多，例如ID3,C4.5等，其使用了信息论中熵的概念。
* 决策树能够实现多分类，能够在较短的时间里对大型数据源做出预测，预测性能较好。
* 缺点
    * 对输入特征要求较高，很多情况下需要做预处理。
    * 识别类别过多时，发生错误的概率较大。
* 决策树的特征选择
    * 输入变量的特征有很多，选择特征作为分类判断的依据之一便是能够具有很好的**区分度**。
    * 那么也就是说，选择出的变量能够更具代表性，以至于区分程度更高，作为决策树的判断节点。

* 决策树生成之ID3算法。决策树生成算法，其对于决策树各个节点应用信息增益准则从而选取特征，在输的每一层进行递归，从而构建整棵树。
* 从根节点开始，在每层选择信息增益最大的作为该节点的判断特征。
* 对所有节点进行相同操作，直到没有特征选择或者所有特征的信息增益均很小为止。

* 决策树的剪枝
    * 决策树是针对训练集进行递归生产的，这样对于训练集效果自然非常好，但是对于未知数据的预测效果可能并不是很好。
    * 即使决策树生成算法生成的决策树模型过于复杂，对未知数据的泛化能力下降，即出现了过拟合的现象。
    * 过拟合是因为树的结构过于复杂，将树的结构精简，就能够减轻过拟合现象，这就是**决策树剪枝**。

* 决策树剪枝算法
    * 决策树从叶节点开始递归地向根节点剪枝
    * 判断一个节点能否被剪掉，只需要比较修剪后和修剪之前的损失函数值大小即可。

* CART算法
    * 分类与回归决策树，其是一个二叉树，根据判断结果划分为“是否”二分类。
    * 决策树生成：基于训练集生成一个尽可能大的决策树。
    * 决策树剪枝：使用验证集对生成的决策树进行剪枝，以便使损失函数最小化。
```
val dt = new DecisionTreeClassifier().setFeaturesCol("features").setLabelCol("label")
val model = dt.fit(train)
val result = model.transform(test)
result.show()
val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")
val accuracy = evaluator.evaluate(result)
println(s"""accuracy is $accuracy""")
```

#### 聚类算法
* 无监督学习、Kmeans\PCA\GMM
* Kmeans算法的描述
    * 设置需要聚类的类别个数K，以及n个训练样本，随机初始化K个聚类中心。
    * 计算每个样本与聚类中心的距离，样本选择最近的聚类中心作为其类别；重新选择聚类中心。
```
val kmeans = new KMeans().setFeaturesCol("features").setK(3).setMaxIter(20)
val model = kmeans.fit(train)
model.transform(test).show()
```

#### 文档主题生成模型
* LDA是一种无监督学习，将主题对应聚类中心，文档作为样本，则LDA也是一种聚类算法。该算法用来将多个文档划分为K个主题，与Kmeans类似。
* LDA是一种基于概率统计的生成算法
* LDA算法是一种常用的主题模型，可以对文档主题进行聚类，同样也可以用在其他的非文档的数据中。
* LDA算法是通过找到词、文档与主题三者之间的统计学关系进行推断的。
* 统计学三大估计方法
    * 最大自然估计logLikelihood,值越大越好
    * 最小二乘估计
    * 贝叶斯估计
