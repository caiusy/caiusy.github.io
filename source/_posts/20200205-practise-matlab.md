---
title: matlab基础
categories: 其他
date: 2020-02-05 00:00:00
tags: MATLAB
  - matlab
---


```
    %%
    % 第一题
    % 给定N和A，N是一个整数，A是一个向量，例如N = 3; A = [ 4 5 6 7]，请使用matlab命令将A中的每一个
    % 元素重复N次，然后形成一个向量，示例计算结果如下：
    N = 3; A = [ 4 5 6 7];
    B=A(ones(1,N),:); % 注意这里下标的实验，可以使用相同的下标多次
    B(:)
```

  
![an1](/images/20200205_practise-matlab_an1.png)  

    
    

```matlab
    %%
    % 假设x是一个向量，例如
    % x = [ 4 4 5 5 5 6 7 7 8 8 8 8 ]
    % 现在我们想得到如下两个向量
    % l = [ 2 3 1 2 4 ]; % x每个元素重复的个数
    % v = [ 4 5 6 7 8 ]; % x中重复元素的值
    x = [ 4 4 5 5 5 6 7 7 8 8 8 8 ]
    i = [find(x(1:end-1) ~= x(2:end)) length(x)]
    l = diff([0 i])
    v = x(i)
```

  
![an2](/images/20200205_practise-matlab_an2.png)  

    
    

```matlab
    %%
    % 求建立以下tables数组
    % table N riqi xianxing Ndanhao Nshuanghao
    % 第一列，行编号
    % 第二列，2017年每天的日期，datetime格式
    % 第三列，如果日期为单号，那么写文本“单号通行”，如果日期是双号，那么写文本“双号通行”，如果是周
    % 末，则写“单双通行”
    % 第四列，写当前日期单号车一共通行了多少天
    % 第五列，写当前日期双号车一共通行了多少天
    NDays = yeardays(2017) %2017年的天数
    N=[1:NDays]';
    riqi=datetime(2017,1,1)+N-1;
    %注意这里，很多时候可以采用先预置一种统一答案，然后其中某些结果可以依次覆盖原有答案
    xianxing = repmat('单号通行',NDays,1);
    a = logical(mod(riqi.Day,2)); %日期是否为单数
    xianxing(~a,:)= repmat('双号通行',sum(~a),1);
    tf = isweekend(riqi);
    xianxing(tf,:) = repmat('单双通行',sum(tf),1);
    idxdanhao=all(xianxing=='单号通行',2)|all(xianxing=='单双通行',2)
    idxshuanghao=all(xianxing=='双号通行',2)|all(xianxing=='单双通行',2)
    Ndanhao=cumsum(idxdanhao);
    Nshuanghao=cumsum(idxshuanghao);
    XianXiangData = table(N,riqi,xianxing,Ndanhao,Nshuanghao);
```

  
![an3](/images/20200205_practise-matlab_an3.png)
    
    

```matlab
    %%
    % 通过load('data.mat')命令载入当前目录下的数据文件，data.mat，然后进行以下处理：
    % data数据是一个手写数字图像的灰度数据，一共2000行785列，2000表示是2000副图片，第一列是当前图片
    % 的数字是几（0-9范围，单个整数），第2列至第785列是图像的灰度数据，图片原始大小为28×28=784个像素
    % 点，然后按照以下编号方式(图片未显示完全)，将784各点转换为了1行，作为了data数据中的一行
    % 现在要求将data第一列提取为一个列向量，命名为trainT，另外第2列至第785列数据转换为28×28×2000的
    % 三维数组，第一页为第一个数字的图像，要求按照图片中的顺序，将第一行中的灰度数据存成新的28×28的数
    % 组，命名为trainD
    data=load('data.mat');
    data1 = data.data(:,2:end);
    trainId = reshape(data1',28,28,2000);
    trainId = permute(trainId,[2 1 3]); %维数的转换
    trianT = data.data(:,1);
    figure
    imshow(uint8(trainId(:,:,7)))
```

  
![an4](/images/20200205_practise-matlab_an4.png)
    
    

```matlab
    %%
    % 在同一个图内绘制两个圆的曲线，一个半径为1，一个半径为2，在右方外侧中部添加图例，“小圆”和”大
    % 圆“，绘图区域设置为正方形，标题设置为“两个圆”，横轴标签为x，纵轴标签为y
    % 双坐标轴图
    % 载入数据datahis.mat，
    % 然后绘制datahis0.t_his为横坐标，datahis0.temp为左侧纵坐标，datahis0.hum为右侧纵坐标的双坐标轴图形
    % 在图像下方外侧添加图例“温度”，“湿度”，横轴标签设置为时间，左侧y轴设置标签为温度，右侧y轴标签
    % 设置为湿度
    % datahis0.t_his为时间，需要转换为datetime类型，然后选取1月10日的数据进行画图
    clear
    figure
    load('datahis.mat')
    t = datetime(datevec(datahis0.t_his));
    idx=t>=datetime(2017,1,10)&t<=datetime(2017,1,11);
    t1 = t(idx);
    temp1 =  datahis0.temp(idx);
    hum1 = datahis0.hum(idx);
    yyaxis left
    plot(t1,temp1)
    xlabel('时间')
    ylabel('温度')
    yyaxis right
    plot(t1,hum1)
    ylabel('温度')
    legend('温度','湿度','Location','southoutside','Orientation','horizontal')
```

  
![an5](/images/20200205_practise-matlab_an5.png)
    
    

```
    %%
    % 多子图的绘制
    % 给定以下数据
    % x = 0:0.01:20; % x坐标
    % y1 = 200*exp(-0.05*x).*sin(x); % Y1
    % y2 = 0.8*exp(-0.5*x).*sin(10*x); % Y2
    % y3 = 100*exp(-0.5*x).*sin(5*x); % Y3
    % y1，y2，y3需要分别绘制一个子图，其中y1占据左侧一半位置，y2占据右侧上方，y3占据右侧下方
    clear
    figure
    x= 0:0.01:20; % x坐标
    y1 = 200*exp(-0.05*x).*sin(x); % Y1
    y2 = 0.8*exp(-0.5*x).*sin(10*x); % Y2
    y3 = 100*exp(-0.5*x).*sin(5*x); % Y3
    subplot(2,2,[1 3])
    plot(x,y1)
    subplot(2,2,2)
    plot(x, y2)
    subplot(2,2,4)
    plot(x,y3)
```

  
![an6](/images/20200205_practise-matlab_an6.png)
    
    

```
    %%
    % 曲面图
    % 绘制函数的网格图,x,y的取值范围为-2到2
    figure
    clear
    x = -2:0.1:2;
    y=x;
    [X,Y]=meshgrid(x,y)
    Z = sin(X.^2+Y.^2);
    mesh(X,Y,Z)
```

  
![an7](/images/20200205_practise-matlab_an7.png)  

    
    

```
    %%
    % 三维饼形图
    % 绘制三维饼形图，各元素所占数值为： [6 3 7 5 1 2 4]，突出显示第1，3个元素，7个元素的标签分别为
    % '周一'到'周日'
    figure
    x =[6 3 7 5 1 2 4];
    labels = {'周一','周二','周三','周四','周五','周六','周日'};
    explode=[1 0 1 0 0 0 0] %突出显示向量x的元素
    pie3(x,explode,labels)
```

  
![an8](/images/20200205_practise-matlab_an8.png)
    
    

```matlab
    %%
    % 绘制如下双纵轴柱形图。数据为：
    % x=1:20;
    % y1=sin(x)+2;
    % y2=(x-10).^2;
    % 注意柱形图等图形也是可以使用双坐标轴绘制的
    clear
    figure
    x=1:20;
    y1=sin(x)+2;
    y2=(x-10).^2;
    yyaxis left
    bar(x+0.2,y1,0.3,'b') %注意图形叠加位置的调整
    ylabel('sin')
    yyaxis right
    bar(x-0.2,y2,0.3,'r')
    ylabel('x^2')
    legend('sin','x^2','Location','southoutside','Orientation','horizontal')
    title('双纵轴柱形图')
    xlabel('x')
```

  
![an9](/images/20200205_practise-matlab_an9.png)
    
    

```bash
    %%
    % 绘制以下数据对应曲线，并增加横轴标签及对应曲线上的数值做标记。并在图中写标记公式文本
    % x=2:20;
    % alpha=x.^2;
    % beta=log(x);
    % y=alpha./beta;
    clear
    figure
    x=2:20;
    alpha=x.^2;
    beta=log(x);
    y=alpha./beta;
    plot(x,y)
    hold on
    plot(sqrt(23),sqrt(23).^2./log(sqrt(23)),'o')
    text(6,100,'$$\frac{\alpha}{\beta}$$','Interpreter','latex');%字符的输出
    xticks(sort([2:2:20 sqrt(23)]))
    h1=gca;
    h1.XTickLabel{3}='$$\sqrt{23}$$'; %字符的输出
    h1.TickLabelInterpreter='latex';
```

  
![an10](/images/20200205_practise-matlab_an10.png)
    
    

```
    %%
    % 绘制如下图形
    % t1 = datetime(2014,1:12,1);
    % temp = [0 2 12 11 15 25 23 27 25 24 12 8];
    clear
    figure
    t1 = datetime(2014,1:12,1);
    temp = [0 2 12 11 15 25 23 27 25 24 12 8];
    h = plot(t1,temp,':*');
    ax = h.Parent;
    title('A Year of Temperatures on the 1st of the Month')
    ylabel('Degrees Celcius ^{\circ}') % 上角标的使用，摄氏度符号
    yt1 = ax.YTickLabel %坐标轴标签为元胞数组，在此基础上进行修改
    ytld = strcat(yt1,'^{\circ}')  %上角标
    ax.YTickLabel = ytld;
```

  
![an11](/images/20200205_practise-matlab_an11.png)
    
    

```
    %%
    % 按照要求绘图
    % t=(pi*(0:1000)/1000)';
    % y=sin(t);
    % 以t为横轴数据，y为纵轴数据绘制图形，要求：
    % 1，将纵坐标轴设置在右方，
    % 2，将横轴标签设置为[0 1/4*pi 1/2*pi 3/4*pi pi]，坐标轴范围设置为0～pi
    % 3，输出图形的figure窗口位置坐标
    % 4，输出图形axes的位置坐标
    clear
    figure
    t=(pi*(0:1000)/1000)';
    y=sin(t);
    h = plot(t,y)
    set(gca,'YAxisLocation','right')
    xticks([0 1/4*pi 1/2*pi 3/4*pi pi])
    xticklabels({'0','1/4 \pi','1/2 \pi','3/4 \pi','\pi'})
    xlim([0 pi])
```

  
![an12](/images/20200205_practise-matlab_an12.png)
    
    

```
    %%
    % 按照以下位置绘制图形
    % 其中需要用到的数据如下，另外图片坐标的位置与示例图中大致相等即可
    % % 图1
    % x=linspace(0.2*pi,20);
    % y=sin(x);
    % % 图2
    % t=0:pi/100:20*pi;
    % x=sin(t);
    % y=cos(t);
    % z=t.*sin(t).*cos(t);
    % % 图3
    % [x,y]=meshgrid(-8:0.5:8);
    % z=sin(sqrt(x.^2+y.^2))./sqrt(x.^2+y.^2+eps);
    clear
    figure
    % 图1
    x=linspace(0.2*pi,20);
    y=sin(x);
    axes('Position',[0.6,0.2,0.2,0.7],'GridLineStyle','-');
    plot(y,x);
    grid on
    % 图2
    axes('Position',[0.1,0.2,0.5,0.5]);
    t=0:pi/100:20*pi;
    x=sin(t);
    y=cos(t);
    z=t.*sin(t).*cos(t);
    plot3(x,y,z);
    % 图3
    axes('Position',[0.1,0.6,0.25,0.3]);
    [x,y]=meshgrid(-8:0.5:8);
    z=sin(sqrt(x.^2+y.^2))./sqrt(x.^2+y.^2+eps);
    mesh(x,y,z)
```

  
![an13](/images/20200205_practise-matlab_an13.png)  

    
    

```
    %%
    % 以下数据进行拟合
    % load ex1.mat
    % cftool
    load ex1.mat
```

  
![an14](/images/20200205_practise-matlab_an14.png)
    
    

```
    %%
    % 使用命令导入文件2016010600.txt中的数据，提取第4列日期数据和第5列时刻数据数据，合并生成一个
    % datetime数组，命名为T
    clear
    a = readtable('2016010600.txt');
    t1 = a.Var4;
    t2 = a.Var5;
    TPvec_y = floor(t1/10000);
    TPvec_mon = floor((t1-TPvec_y*10000)/100);
    TPvec_d = mod(t1,100)
    TPvec_h = floor(t2/100);
    TPvec_min=mod(t2,100);
    TPvec_s = zeros(size(t1,1),1);
    t_vec=[TPvec_y TPvec_mon TPvec_d TPvec_h TPvec_min TPvec_s];
    T = datetime(t_vec)
```

  
![an15](/images/20200205_practise-matlab_an15.png)
    
    

```
    %%
    % 生成与2016010600.txt文件中第6列数据概率分布相同的随机数，数量10000个.
    clear
    figure
    a = readtable('2016010600.txt')
    v = a.Var6;
    histogram(v,20)
    [N,edges] = histcounts(v,20);
    rm = RDrnd(N,10000);
    figure
    histogram(rm,20)
```

  
![an16](/images/20200205_practise-matlab_an16.png)
