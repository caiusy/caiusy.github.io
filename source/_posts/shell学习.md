---
title: shell学习
categories: 技术
typora-copy-images-to: ./shell学习
date: 2023-01-22 16:55:53
tags: 技术
---
## shell学习

### 1. shell概述

Shell 是一个命令解释器， 用于接收应用程序/用户命令， 然后调用操作系统内核。

![image-20220710001433868](./images/image-20220710001433868.png)

 shell 还是一个功能相当强大的编程语言， 易编写，易调试，灵活性强。

### 2. shell解析器

~~~sh
/bin/sh
/bin/bash
/usr/bin/bash
/bin/rbash
/usr/bin/rbash
/usr/bin/sh
/bin/dash
/usr/bin/dash
~~~

~~~shel
root@2a63e139ac66:/bin# echo $SHELL
/bin/bash
~~~

系统默认的是bash

### 3. Shell 脚本入门

#### 3.1 脚本格式

脚本以#!/bin/bash 开头（指定解析器） 

#### 3.2 第一个shell脚本： helloworld

##### 3.2.1 需求： 创建一个shell脚本， 输出helloworld

~~~shell
#!/bin/bash
echo "helloworld shell"
~~~

~~~sh
 cd shelldata
# ls
# pwd
/root/shelldata
# touch helloworld.sh
# vim helloworld.sh

#
#
# sh helloworld.sh
helloworld shell
~~~

权限不够

~~~
# ./ helloworld.sh
/bin/sh: 33: ./: Permission denied
~~~

##### 3.2.2 脚本的常用执行方式

第一种： 采用bash或者sh+脚本的相对路径或者绝对路径（不用赋予脚本+x权限）

第二种： 采用输入脚本的绝对路径或者相对路径执行脚本（必须具有可执行权限+x）

> 注意： 第一种执行方法， 本质是bash解析器帮你执行脚本， 所以脚本本身不需要执行权限， 第二种执行方法， 本质是脚本自己执行，所以需要执行权限。

#### 3.3 第二个shell脚本： 多命令处理

##### 3.3.1 需求：

在/home/atguigu/目录下创建一个bangzhang.txt, 在banzhang.txt中增加“I Love  cls"

##### 3.3.2 案例实操

~~~shell
#!/bin/bash
cd /root/shelldata/
touch banzhang.txt
echo "I Love cls">> banzhang.txt
~~~

结果显示

~~~shell
sh-5.1# cat banzhang.txt
I Love cls
I Love cls
sh-5.1# rm banzhang.txt
sh-5.1# bash hadoop101.sh
sh-5.1# cat banzhang.txt
I Love cls
sh-5.1# vim hadoop101.sh
~~~

### 4. Shell中的变量

#### 4.1 系统变量

##### 4.1.1 常用系统变量

$HOME、$PWD、$SHELL、$USER等

##### 4.1.2 案例实操

查看系统变量的值

~~~shell
sh-5.1# echo $HOME
/root
sh-5.1# echo $PWD
/root/shelldata
sh-5.1#
~~~

显示当前shell中所有变量

~~~shell
BASH=/bin/sh
BASHOPTS=checkwinsize:cmdhist:complete_fullquote:expand_aliases:extquote:force_fignore:globasciiranges:hostcomplete:interactive_comments:progcomp:promptvars:sourcepath
BASH_ALIASES=()
BASH_ARGC=()
BASH_ARGV=()
BASH_CMDS=()
BASH_LINENO=()
BASH_SOURCE=()
BASH_VERSINFO=([0]="5" [1]="1" [2]="16" [3]="1" [4]="release" [5]="x86_64-pc-linux-gnu")
BASH_VERSION='5.1.16(1)-release'
COLUMNS=177
DIRSTACK=()
EUID=0
GROUPS=()
HISTFILE=/root/.bash_history
HISTFILESIZE=500
HISTSIZE=500
HOME=/root
HOSTNAME=8b25e507bcca
HOSTTYPE=x86_64
IFS='
'
LINES=50
MACHTYPE=x86_64-pc-linux-gnu
MAILCHECK=60
OLDPWD=/root
OPTERR=1
OPTIND=1
OSTYPE=linux-gnu
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
PIPESTATUS=([0]="0")
POSIXLY_CORRECT=y
PPID=0
PS1='\s-\v\$ '
PS2='> '
PS4='+ '
PWD=/root/shelldata
SHELL=/bin/bash
SHELLOPTS=braceexpand:emacs:hashall:histexpand:history:interactive-comments:monitor:posix
SHLVL=1
TERM=xterm
UID=0
_=/root/shelldata
~~~

#### 4.2 自定义变量

##### 4.2.1 基本语法

定义变量： 变量=值

~~~shell
sh-5.1# A=1
sh-5.1# echo A
A
sh-5.1# echo $A
1
~~~

撤销变量： unset变量

~~~shell
sh-5.1# unset A
sh-5.1# echo $A

sh-5.1#
~~~

声明一个静态变量： readonly变量， 注意：不能unset

~~~shell
sh-5.1# readonly B=3
sh-5.1# echo $B
3
sh-5.1# unset B
sh: unset: B: cannot unset: readonly variable
sh-5.1#
~~~

##### 4.2.2 变量定义规则

1. 变量名称可以由字母、数字和下划线构成， 但是不能以数字开头， 环境变量名建议大写
2. 等号两侧不能有空格
3. 在bash中， 变量默认类型都是字符串类型， 无法直接进行数值运算。

~~~shell
sh-5.1# C=1+1
sh-5.1#
Display all 505 possibilities? (y or n)
sh-5.1# echo $C
1+1
sh-5.1#
~~~

4. 变量的值如果有空格，需要用双引号或者单引号括起来。

~~~shell
sh-5.1# D=bangzhang love mm
sh: love: command not found
sh-5.1# D='bangzhang love mm'
sh-5.1# echo $D
bangzhang love mm
sh-5.1#
~~~

5. 可以把变量提升为全局环境变量， 可供其他shell程序使用

export 变量名

~~~shell
sh: ./helloworld.sh: Permission denied
sh-5.1# chmod +x helloworld.sh
sh-5.1# ./helloworld.sh
helloworld shell

sh-5.1# export D
sh-5.1# ./helloworld.sh
helloworld shell
bangzhang love mm
~~~

#### 4.3 特殊变量：$n

##### 4.3.1 基本语法

$n（功能描述：n为数字，$0代表该脚本名称，$1-$9代表第一到第九个参数， 十以上的参数需用大括号包含，如${10}）

##### 4.3.2  案例实操

1. 输出该脚本文件名称、输入参数1和输入参数2的值

~~~shell
#!/bin/bash

echo "$0 $1 $2"

~                                                                                                           sh-5.1# vim parameter.sh
sh-5.1# bash parameter.sh
parameter.sh
sh-5.1# bash parameter.sh banzhang
parameter.sh banzhang
sh-5.1# bash parameter.sh banzhang lobve
parameter.sh banzhang lobve
sh-5.1# bash parameter.sh banzhang lobve mm
parameter.sh banzhang lobve
sh-5.1# vim parameter.sh              ~       
~~~

#### 4.4 特殊变量：$#

##### 4.4.1 基本语法

$# (功能描述： 获取所有输入参数的个数， 常用于循环）

~~~shell
#!/bin/bash

echo "$0 $1 $2"
echo $#
~        
·································
sh-5.1# vim parameter.sh
sh-5.1# ls
banzhang.txt  hadoop101.bahs  hadoop101.sh  helloworld.sh  parameter.sh
sh-5.1# chmod +x helloworld.sh
sh-5.1# chmod 777 parameter.sh
sh-5.1# ./parameter.sh cls xyz 111
./parameter.sh cls xyz
3
sh-5.1#

~~~

#### 4.5 特殊变量：$*、$@

$* （功能描述： 这个变量代表命令行中所有的参数，$*把所有的参数看成一个整体）

$@（功能描述： 这个变量也代表命令行中所有的参数， 只不过$@把每个参数区分对待）

~~~shell
#!/bin/bash

echo "$0 $1 $2"
echo $#
echo $*
echo $@
····························
sh-5.1# ./parameter.sh banhh lobe  1111
./parameter.sh banhh lobe
3
banhh lobe 1111
banhh lobe 1111
~~~

#### 4.6 特殊变量$?

$? (功能描述： 最后一次执行的命令的返回状态。如果这个变量的值为0表示上个命令执行正确；如果为非零（具体哪个值，由命令自己决定)，证明上一个命令执行不正确了。

~~~shell
证明helloworld脚本 是否正确执行
sh-5.1# ./helloworld.sh
helloworld shell

sh-5.1# echo $?
0
sh-5.1#
~~~

### 第5章 运算符

#### 5.1 基本语法

“$((运算式))” 或“$[运算式]”

expr +,- ,\\*,/,% 加、减、乘、除，取余

注意： expr运算符间要有空格

1. 计算3+2的值

~~~shell
sh-5.1# expr 2 + 3
5
~~~

2. 计算3-2 的值

~~~shell
sh-5.1# expr 3 - 2
1
sh-5.1#
~~~

3. 计算（2+3）*4的值

~~~shell
sh: 1: command not found
sh-5.1# expr 'expr 2 + 3' \* 4

expr: non-integer argument

采用 $[运算式]方式
sh-5.1#  s=$[(2+3)*4]
sh-5.1# echo $s
20
~~~

### 第六章 条件判断

#### 6.1 基本语法

[ condition ]  （注意 condition 前后要有空格）

注意： 条件非空即为true， [ atguigu ] 返回true， []返回false

#### 6.2 常用判断条件

1. 两个整数之间比较

   = 字符串比较

   -lt 小于（less than)                                        -le 小于等于(less equal)

   -eq 等于（equal）                                        -gt 大于 (greater than)

   -ge 大于等于（greater equal）                   -ne 不等于(Not equal)

2. 按照文件权限进行判断

​    -r 有读的权限(read)                                            -w 有写的权限(write)

​    -x 有执行的权限(execute)                

3. 按照文件类型进行判断

   -f  文件存在 并且是一个常规的文件(file)

   -e 文件存在（existence）                                     -d 文件存在并且是一个目录(directory)

~~~shell
23 是否大于等于22
sh-5.1# [ 23 -ge 22 ]
sh-5.1# echo $?
0

helloworld.sh 是否具有写权限
sh-5.1# [ -w helloworld.sh ]
sh-5.1# echo $?
0
~~~

4. 多条件判断(&& 表示前面一条命令执行成功时候，才执行后一条命令， || 表示上一条命令执行失败后，才执行下一条命令)

~~~shell
sh-5.1# [ condition ] && echo OK || echo notok
OK
sh-5.1# [  ] && echo OK || echo notok
notok
~~~

#### 第七章 流程控制

##### 7.1 if 判断

if [ 条件判断式 ]:then

​     程序

fi

或者

if [ 条件判断式 ]

​    then

​            程序

fi

注意事项：

1. [ 条件判断式 ]， 中括号和条件判断式之间必须有空格
2. if 后要有空格

~~~shell
输入一个数字， 如果是1 ， 则输出banzhang zhenshuai， 如果是2，则输出shell mei，如果是其他，则什么也不输出

#!/bin/bash

if [ $1 -eq 1 ]
then
        echo "banzhangzhenshuai"

elif [ $1 -eq 2 ]
then
        echo "shell mei"

fi

sh-5.1# bash if.sh
if.sh: line 3: [: -eq: unary operator expected
if.sh: line 7: [: -eq: unary operator expected
sh-5.1# bash if.sh 1
banzhangzhenshuai
sh-5.1# bash if.sh \2
shell mei
sh-5.1# bash if.sh 2
shell mei
~~~


##### 7.2 case 断句

case $变量名 in

“值1”）

​       如果变量的值等于1，则执行程序1

   ;;

“值2”）

​       如果变量的值等于2，则执行程序2

   ;;

...省略其他分支...

*）

如果变量的值都不是以上的值，则执行次程序

;;

esac

注意事项：

1. case行结尾必须为单词“in” ， 每一个模式匹配必须以右括号“）” 结束。
2. 双分号“;;”表示命令序列结束，相当于java中的break
3. 最后的“*）”表示默认模式， 相当于java中的default

~~~shell
输入一个数字，如果是1 则输出bangzag ，如果是2 则输出java，如果是其他则输出python
#!/bin/bash

case $1 in
1) echo "c++"
;;
2) echo "java"
;;
*) echo "python"
;;
esac


sh-5.1# vim case.sh
sh-5.1# bash case.sh 2
case.sh: line 10: syntax error: unexpected end of file
sh-5.1# vim case.sh
sh-5.1# bash case.sh 2
java
sh-5.1# bash case.sh 3
python
sh-5.1# bash case.sh 1
c++
~~~

##### 7.3 for循环

for (( 初始值;循环控制条件;变量变化 ))

   do

​        程序

   done

~~~~shell
从1 加到100
#!/bin/bash
s=0
for(( i=0;i<=100;i++ ))
do
        s=$[$s+$i]
done
echo $s

sh-5.1# vim for1.sh
sh-5.1# bash for1.sh
5050
~~~~

语法2

for 变量 in 值1 值2 值3...

do 

​       程序

done

~~~shell
打印所有输入参数
#!/bin/bash
for i in $*
do
        echo "shell love $i"
done
for k in "$*"
do
                echo "shell love $k"
done

for j in "$@"
do
        echo "shell love $j"
done
sh-5.1# touch for2.sh
sh-5.1# vim for2.sh
sh-5.1# sh for2.sh 1 2 3
shell love 1
shell love 2
shell love 3sh-5.1# sh for2.sh 1 2 3
shell love 1
shell love 2
shell love 3
shell love 1 2 3
shell love 1
shell love 2
shell love 3
~~~

##### 7.4 while循环

while [ 条件判断式 ]

do

   程序

done

~~~shell
从1+到100
#!/bin/bash

s=0
i=1
while [ $i -le 100 ]
do
        s=$[$s + $i]
        i=$[$i + 1]
done
echo $s
~        
sh-5.1# bash while.sh
5050
~~~

#### 第八章 read 读取控制台输入

read(选项)(参数)

选项：

-p： 指定读取值时的提示符：

-t :   指定读取值时等待的时间（秒）

参数

​       变量：指定读取值得变量名

1. 提示7s内读取控制台输入的名称

~~~shell
#!/bin/bash

read -t 7 -p "Enter your name i 7 seconds " NAME
echo $NAME
~          
sh-5.1# bash read.sh shell
Enter your name i 7 seconds shell
shell

~~~

#### 第九章 函数

##### 9.1 系统函数

basename基本语法

basename [string/pathname] [suffix] (功能描述： basename 的命令会删掉所有前缀包括最后一个（'/')字符，然后将字符串显示出来

选项：

suffix为后缀， 如果suffix被指定了， basenname会将pathname或者string中的suffix去掉。

~~~shell
截取该/homeshell/banzhang.txt路径的文件名称
sh-5.1# basename /homeshell/banzhang.txt .txt
banzhang
~~~

dirname   文件绝对路径 ( 功能描述： 从给定的包含绝对路径中取出文件名（非目录的部分)，然后返回剩下的路径)

~~~shell
获取截取该/homeshell/banzhang.txt路径的文件路径
sh-5.1# dirname /homeshell/banzhang.txt
/homeshell
~~~

##### 9.2 自定义函数

基本语法

[ function ] funname[()]

{

​		Action；

​		[return int;]

}

2. 经验技巧

2.1 必须在调用地方之前，先声明函数， shell脚本是逐行运行的， 不会像其他语言一样 先编译。

2.2 函数返回值， 只能通过$?系统变量或得，可以显示加:return 返回，如果不加，将以最后一条命令运行结果作为返回值，return后跟数值n(0-255)

~~~shell
计算两个输入参数的和
#!/bin/bash

function sum()
{
        s=0;
        s=$[$1+$2]
        echo $s
}
read -p "input your paramter1: " P1
read -p "input your paramter1: " P2
sum $P1 $P2

sh-5.1# vim sum.sh
sh-5.1# bash sum.sh
input your paramter1:
input your paramter1:
sum.sh: line 6: +: syntax error: operand expected (error token is "+")
sh-5.1# bash sum.sh
input your paramter1: 10
input your paramter1: 20
30
~~~


#### 第十章 shell工具（重点）

##### 10.1 cut

cut的工作就是“剪”， 具体的说就是在文件中负责剪切数据用的。cut命令从文件的每一行剪切字节、字符和字段并将这些值输出

1. 基本用法

​     cut[选项参数]  filenames

​    说明: 默认分割符是制表符

2. 选项参数说明

![image-20220710225214935](./images/image-20220710225214935.png)

~~~shell
数据准备
sh-5.1# touch cut.txt
sh-5.1# vim cut.txt
sh-5.1# cat cut.txt
dong sheng
guan zheng
wo wo
la   la
lei lei

sh-5.1# cut -d " " -f 1 cut.txt
dong
guan
wo
la
lei

在cut文件中切割出guan
h-5.1# cat cut.txt | grep guan
guan zheng
sh-5.1# cat cut.txt | grep guan | cut -d " " -f 1
guan


选取系统PATH变量值， 第二个“：”开始后的所有路径：
sh-5.1# echo $PATH
/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
sh-5.1# echo $PATH | cut -d : -f 3-
/usr/sbin:/usr/bin:/sbin:/bin
切割ifconfig后打印的ip地址
失败了。
sh-5.1# ifconfig eth0 | grep "inet" | cut -d '' -f 2
        inet 172.17.0.2  netmask 255.255.0.0  broadcast 172.17.255.255
sh-5.1#
~~~

![image-20220710230520164](./images/image-20220710230520164.png)

##### 10.2 sed

  sed是一种流编辑器， 它一次处理一行内容， 处理时，吧当前处理的行存储在临时缓存区中， 称为：模式空间，接着用sed命令处理缓冲区中的内容， 处理完成后，把缓冲区的内容送往屏幕， 接着处理下一行， 这样不断重复， 直到文件末尾。文件内容没有改变， 除非你使用重定向存储输出。

基本用法

sed[选项参数] “command”  filename

![image-20220710230902184](./images/image-20220710230902184.png)

#### 10.2.2 实战

~~~shell
将mei nv 这个单词插入到sed.txt 第二行下， 打印
sh-5.1# touch sed.txt
sh-5.1# vim sed.txt
sh-5.1# sed '2a mei nv' sed.txt
dong shen
ot python
mei nv
ll ll
ko ok
删除文件中包含所有wo的行
sh-5.1# sed "/m/e" sed.txt
dong shen
ot python
ll ll
ko ok
将sed中wo替换为ni
sh-5.1# sed "s/ot/to/g" sed.txt
dong shen
to python
ll ll
ko ok
注意：
g表示global ，全部替换

将sed中第二行删除并将wo替换为ni
sh-5.1# sed -e "2d" -e "s/to/ot/g" sed.txt
dong shen
ll ll
ko ok
~~~

#### 10.3 awk

一个强大的文本分析工具， 把文件逐行的读入，以空格为默认分隔符将每行切片， 切开的部分再进分析处理。

1. 基本用法
2. awk [选项参数] ‘pattern1{action1} pattern2{action2} ....' filename
3. pattern: 表示AWK在数据中查找的内容，就是匹配模式
4. action:在找到匹配时，所执行的一系列命令

![image-20220710231854660](./images/image-20220710231854660.png)

##### 实操

数据准备

~~~
sh-5.1# cp /etc/passwd ./
sh-5.1# ls
banzhang.txt  case.sh  cut.txt  for1.sh  for2.sh  hadoop101.bahs  hadoop101.sh  helloworld.sh  if.sh  parameter.sh  passwd  read.sh  sed.txt  sum.sh  while.sh
sh-5.1# awk -F: '/^root/{print $7}' passwd /bin/bash
/bin/bash
sh-5.1#

~~~

1. 搜索passwd文件以root关键词开头的所有行，并输出该行的第7列

2. 搜索passwd文件以root关键词开头的所有行，并输出该行的第7列和第一列，中间以“，” 分隔

   ~~~shell
   sh-5.1# awk -F: '/^root/{print  $1“，”$7}' passwd /bin/bash
   ~~~

   注意只有匹配了pattern的行才会执行action

3. 只显示、etc/passwd的第一列和第七列，以逗号分隔，且在所有行前面铁建列名user， shell在最后一行叠加“dahaige , /bin/zuishuai".

     ~~~shell
     awk -F: 'BEGIN{print "user,shell"}{print $1","$7} END{print "dahaiuge,/bin/zuishuai"}' passwd
     ~~~

注意LBEGIN在所有数据读取行之前执行， END在所有数据执行完之后执行

4. 将passwd中的用户id增加数值1并输出

   ~~~shell
   awk -vi1-F: '{print $3+i}' passwd
   ~~~

##### awk的内置变量

![image-20220710232835635](./images/image-20220710232835635.png)

##### 实操

1. 统计passwd文件名，每行的行号，每行的列数

~~~shell
awk -F: '{print "filename:" FILENAME ", linenumer:" NR ",columns:" NF}' passwd
~~~

2. 切割ip

   ~~~shell
   ifconfig eth0 | grep "inet addr" | awk -F : '{print $2}' | awk -F " " '{print $1}'
   ~~~

3. 查询sed.txt中空行所在的行号

   ~~~shell
   awk '/^$/{print NR}' sed.txt
   ~~~

   

#### 10.4 sort

sort 命令在LInux里非常游泳，它将文件排序，并将排序结果标准输出

![image-20220711002337565](./images/image-20220711002337565.png)

##### 实操

1. 按照“：” 分隔之后排序

   ~~~shell
   sort-t ： -nrk 2 sort.sh
   ~~~

#### 第十一章 企业面试题

#####   11.1 京东

问题1 ：使用linux命令查询file1中空行所在的行号

答案：

~~~shell
awk '/^$/{print NR}' sed.txt
~~~

 问题2： 有文件chengji.txt内容如下：

张三 40

李四 50

王五 60

使用linux命令计算第二列的和并输出

~~~
cat chenji.TXT| AWK -F " " '{sum+=$2} END{print sum}'
~~~

##### 11.2 搜狐&和讯网

问题1： shell脚本里如何检查一个文件是否存在，如果不存在该如何处理？

~~~shell
#!/bin/bash

if [ -f file.txt ]; then
   echo "文件存在！"
else
   echo "文件不存在！"
fi
~~~

##### 11.3 新浪

问题1： 用shell写一个脚本，对文本中无序的一列数据排序

~~~shell
sort -n  test.txt|awk '{a+=$0;print$0}END{print "SUM-"a}'
~~~

##### 11.4 金和网络

问题1： 请用shell脚本写出 查找当前文件夹，(/home) 下所有文本文件内容包含有字符“shen”的文件名称

~~~shell
grep -r "shen" /home |cut -d ":" -f 1 <留下文件名>
~~~

