---
title: kmeans算法
date: 2019-09-07 00:00:00
categories:
  - 深度学习
tags:
  - Python
---
* * *  
  
K-Means的思想十分简单，首先随机指定类中心，根据样本与类中心的远近划分类簇，接着重新计算类中心，迭代直至收敛。但是其中迭代的过程并不是主观地想象得出，事实上，若将样本的类别看做为“隐变量”（latent variable），类中心看作样本的分布参数，这一过程正是通过EM算法的两步走策略而计算出，其根本的目的是为了最小化平方误差函数E。

## kmeans算法的最大弱点：只能处理球形的簇（理论）

## kmeans 计算步骤

  * 1.随机选取K个聚类中心，这里的k值可以自己设定
  * 2.先设置一个聚类标志，用来保存当前的 样本与第几个聚类中心最近
  * 3.计算每个样例与每个聚类中心的距离，保存最小距离的k以及距离
  * 4.更新聚类中心，为当前类别所有样本的均值大小![](/2019/09/07/kmeans算法/images/20190907_kmeans算法_myplot.png)
        
        1  
        2  
        3  
        4  
        5  
        6  
        7  
        8  
        9  
        10  
        11  
        12  
        13  
        14  
        15  
        16  
        17  
        18  
        19  
        20  
        21  
        22  
        23  
        24  
        25  
        26  
        27  
        28  
        29  
        30  
        31  
        32  
        33  
        34  
        35  
        36  
        37  
        38  
        39  
        40  
        41  
        42  
        43  
        44  
        45  
        46  
        47  
        48  
        49  
        50  
        51  
        52  
        53  
        54  
        55  
        56  
        57  
        58  
        59  
        60  
        61  
        62  
        63  
        64  
        65  
        66  
        67  
        68  
        69  
        70  
        71  
        72  
        73  
        74  
        75  
        76  
        77  
        78  
        79  
        80  
        81  
        82  
        83  
        84  
        85  
        86  
        87  
        88  
        89  
        90  
        91  
        92  
        

| 
        
        from numpy import *  
        import time  
        import matplotlib.pyplot as plt  
          
          
        # calculate Euclidean distance  
        def euclDistance(vector1, vector2):  
            return sqrt(sum(power(vector2 - vector1, 2)))  # 求这两个矩阵的距离， vector1, vector2 均为矩阵  
          
          
        # init centroids with random samples  
        # 在样本集中随机选取k个样本点作为初始质心  
          
          
        def initCentroids(dataSet, k):  
            numSamples, dim = dataSet.shape  # 矩阵的行数、列数  
            centroids = zeros((k, dim))  # 感觉要不要你都可以  
            for i in range(k):  
                index = int(random.uniform(0, numSamples))  # 随机产生一个浮点数，然后将其转化为int型  
                centroids[i, :] = dataSet[index, :]  
            return centroids  
          
        # k-means cluster  
        # dataSet为一个矩阵  
        # k为将dataSet矩阵中的样本分成k个类  
        def kmeans(dataSet, k):  
            numSamples = dataSet.shape[0]  # 读取矩阵dataSet的第一维度的长度,即获得有多少个样本数据  
            # first column stores which cluster this sample belongs to,  
            # second column stores the error between this sample and its centroid  
            clusterAssment = mat(zeros((numSamples, 2)))  # 得到一个N*2的零矩阵  
            clusterChanged = True  
          
            ## step 1: init centroids  
            centroids = initCentroids(dataSet, k)  # 在样本集中随机选取k个样本点作为初始质心  
          
            while clusterChanged:  
                clusterChanged = False  
                ## for each sample  
                for i in range(numSamples):  # range  
                    minDist = 100000.0  
                    minIndex = 0  
                    ## for each centroid  
                    ## step 2: find the centroid who is closest  
                    # 计算每个样本点与质点之间的距离，将其归内到距离最小的那一簇  
                    for j in range(k):  
                        distance = euclDistance(centroids[j, :], dataSet[i, :])  
                        if distance < minDist:  
                            minDist = distance  
                            minIndex = j  
          
                            ## step 3: update its cluster  
                    # k个簇里面与第i个样本距离最小的的标号和距离保存在clusterAssment中  
                    # 若所有的样本不在变化，则退出while循环  
                    if clusterAssment[i, 0] != minIndex:  
                        clusterChanged = True  
                        clusterAssment[i, :] = minIndex, minDist ** 2  # 两个**表示的是minDist的平方  
          
                ## step 4: update centroids  
                for j in range(k):  
                    # clusterAssment[:,0].A==j是找出矩阵clusterAssment中第一列元素中等于j的行的下标，返回的是一个以array的列表，第一个array为等于j的下标  
                    pointsInCluster = dataSet[nonzero(clusterAssment[:, 0].A == j)[0]]  # 将dataSet矩阵中相对应的样本提取出来  
                    centroids[j, :] = mean(pointsInCluster, axis=0)  # 计算标注为j的所有样本的平均值  
          
            print('Congratulations, cluster complete!')  
            return centroids, clusterAssment  
          
          
        # show your cluster only available with 2-D data  
        # centroids为k个类别，其中保存着每个类别的质心  
        # clusterAssment为样本的标记，第一列为此样本的类别号，第二列为到此类别质心的距离  
        def showCluster(dataSet, k, centroids, clusterAssment):  
            numSamples, dim = dataSet.shape  
            if dim != 2:  
                print("Sorry! I can not draw because the dimension of your data is not 2!")  
                return 1  
          
            mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']  
            if k > len(mark):  
                print("Sorry! Your k is too large! please contact wojiushimogui")  
                return 1  
          
                # draw all samples  
            for i in range(numSamples):  
                markIndex = int(clusterAssment[i, 0])  # 为样本指定颜色  
                plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])  
          
            mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']  
            # draw the centroids  
            for i in range(k):  
                plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=12)  
          
            plt.show()  
          
  
---|---  

    
    
    1  
    2  
    3  
    4  
    5  
    6  
    7  
    8  
    9  
    10  
    11  
    12  
    13  
    14  
    15  
    16  
    17  
    18  
    19  
    20  
    21  
    22  
    23  
    24  
    25  
    26  
    27  
    

| 
    
    
    from numpy import *  
    import time  
    import matplotlib.pyplot as plt  
    import kmeans  
      
    ## step 1: load data  
    print("step 1: load data...")  
    dataSet = []  # 列表，用来表示，列表中的每个元素也是一个二维的列表；这个二维列表就是一个样本，样本中包含有我们的属性值和类别号。  
    # 与我们所熟悉的矩阵类似，最终我们将获得N*2的矩阵，每行元素构成了我们的训练样本的属性值和类别号  
    fileIn = open("./testSet.txt")  # 是正斜杠  
    for line in fileIn.readlines():  
        temp = []  
        lineArr = line.strip().split('\t')  # line.strip()把末尾的'\n'去掉  
        temp.append(float(lineArr[0]))  
        temp.append(float(lineArr[1]))  
        dataSet.append(temp)  
    # dataSet.append([float(lineArr[0]), float(lineArr[1])])  
    fileIn.close()  
    ## step 2: clustering...  
    print("step 2: clustering...")  
    dataSet = mat(dataSet)  # mat()函数是Numpy中的库函数，将数组转化为矩阵  
    k = 4  
    centroids, clusterAssment = kmeans.kmeans(dataSet, k)  # 调用KMeans文件中定义的kmeans方法。  
      
    ## step 3: show the result  
    print("step 3: show the result...")  
    kmeans.showCluster(dataSet, k, centroids, clusterAssment)  
      
  
---|---  
      
    
    1  
    2  
    3  
    4  
    5  
    6  
    7  
    8  
    9  
    10  
    11  
    12  
    13  
    14  
    15  
    16  
    17  
    18  
    19  
    20  
    21  
    22  
    23  
    24  
    25  
    26  
    27  
    28  
    29  
    30  
    31  
    32  
    33  
    34  
    35  
    36  
    37  
    38  
    39  
    40  
    41  
    42  
    43  
    44  
    45  
    46  
    47  
    48  
    49  
    50  
    51  
    52  
    53  
    54  
    55  
    56  
    57  
    58  
    59  
    60  
    61  
    62  
    63  
    64  
    65  
    66  
    67  
    68  
    69  
    70  
    71  
    72  
    73  
    74  
    75  
    76  
    77  
    78  
    79  
    80  
    

| 
    
    
    1.658985	4.285136    
    -3.453687	3.424321    
    4.838138	1.151539    
    -5.379713	-3.362104    
    0.972564	2.924086    
    -3.567919	1.531611    
    0.450614	-3.302219    
    -3.487105	-1.724432    
    2.668759	1.594842    
    -3.156485	3.191137    
    3.165506	-3.999838    
    -2.786837	-3.099354    
    4.208187	2.984927    
    -2.123337	2.943366    
    0.704199	-0.479481    
    -0.392370	-3.963704    
    2.831667	1.574018    
    -0.790153	3.343144    
    2.943496	-3.357075    
    -3.195883	-2.283926    
    2.336445	2.875106    
    -1.786345	2.554248    
    2.190101	-1.906020    
    -3.403367	-2.778288    
    1.778124	3.880832    
    -1.688346	2.230267    
    2.592976	-2.054368    
    -4.007257	-3.207066    
    2.257734	3.387564    
    -2.679011	0.785119    
    0.939512	-4.023563    
    -3.674424	-2.261084    
    2.046259	2.735279    
    -3.189470	1.780269    
    4.372646	-0.822248    
    -2.579316	-3.497576    
    1.889034	5.190400    
    -0.798747	2.185588    
    2.836520	-2.658556    
    -3.837877	-3.253815    
    2.096701	3.886007    
    -2.709034	2.923887    
    3.367037	-3.184789    
    -2.121479	-4.232586    
    2.329546	3.179764    
    -3.284816	3.273099    
    3.091414	-3.815232    
    -3.762093	-2.432191    
    3.542056	2.778832    
    -1.736822	4.241041    
    2.127073	-2.983680    
    -4.323818	-3.938116    
    3.792121	5.135768    
    -4.786473	3.358547    
    2.624081	-3.260715    
    -4.009299	-2.978115    
    2.493525	1.963710    
    -2.513661	2.642162    
    1.864375	-3.176309    
    -3.171184	-3.572452    
    2.894220	2.489128    
    -2.562539	2.884438    
    3.491078	-3.947487    
    -2.565729	-2.012114    
    3.332948	3.983102    
    -1.616805	3.573188    
    2.280615	-2.559444    
    -2.651229	-3.103198    
    2.321395	3.154987    
    -1.685703	2.939697    
    3.031012	-3.620252    
    -4.599622	-2.185829    
    4.196223	1.126677    
    -2.133863	3.093686    
    4.668892	-2.562705    
    -2.793241	-2.149706    
    2.884105	3.043438    
    -2.967647	2.848696    
    4.479332	-1.764772    
    -4.905566	-2.911070  
      
  
---|---  
  
参考链接：<https://github.com/wojiushimogui/kmeans>
