---
title: Python：logging模块
date: 2019-11-04 00:00:00
---


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
    

| 
    
    
      
    In [12]: import logging  
        ...:  
        ...: logging.basicConfig(level=logging.DEBUG,  
        ...:                     filename='output.log',  
        ...:                     datefmt='%Y/%m/%d %H:%M:%S',  
        ...:                     format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')  
        ...: logger = logging.getLogger(__name__)  
        ...:  
        ...: logger.info('This is a log info')  
        ...: logger.debug('Debugging')  
        ...: logger.warning('Warning exists')  
        ...: logger.info('Finish')  
      
  
---|---  
  
2019-11-04 13:00:45,976 - **main** \- INFO - This is a log info  
2019-11-04 13:00:45,977 - **main** \- WARNING - Warning exists  
2019-11-04 13:00:45,977 - **main** \- INFO - Finish

设置level等级，从而控制log输出的级别。
    
    
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
    

| 
    
    
    In [13]: import logging  
        ...:  
        ...: logging.basicConfig(level=logging.DEBUG,  
        ...:                     filename='output.log',  
        ...:                     datefmt='%Y/%m/%d %H:%M:%S',  
        ...:                     format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')  
        ...: logger = logging.getLogger(__name__)  
        ...: logger.setLevel(level=logging.DEBUG)  
        ...: logger.info('This is a log info')  
        ...: logger.debug('Debugging')  
        ...: logger.warning('Warning exists')  
        ...: logger.info('Finish')  
      
  
---|---  
  
如果不设置logger的Level的话， debug’的信息也不会被输出。

需要设置 logger.setLevel(level=logging.DEBUG)，然后信息就可以正常的显示出来了。

2019-11-04 13:10:01,634 - **main** \- INFO - This is a log info  
2019-11-04 13:10:01,634 - **main** \- DEBUG - Debugging  
2019-11-04 13:10:01,635 - **main** \- WARNING - Warning exists  
2019-11-04 13:10:01,639 - **main** \- INFO - Finish

CSDN博客地址：  
<https://blog.csdn.net/eilot_c/article/details/102894687>
