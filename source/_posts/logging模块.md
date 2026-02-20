---
title: logging模块
date: 2019-08-28 00:00:00
categories:
  - Python
tags:
  - 编程
---
## 简单使用
    
    
    1  
    2  
    3  
    4  
    5  
    6  
    7  
    

| 
    
    
    import logging  
      
    logging.debug("debug msg")  
    logging.info("info msg")  
    logging.warn("warn msg")  
    logging.error("error msg")  
    logging.critical("critical msg")  
      
  
---|---  
  
默认情况下, logging模块将日志打印到屏幕上, 只有日志级别高于WARNING的日志信息才回输出
