---
title: xml解析
date: 2019-09-07 00:00:00
categories:
  - 算法
tags:
  - Python
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
    

| 
    
    
    <annotation>  
    	<folder>ILSVRC2015_VID_train_0002/ILSVRC2015_train_00555002</folder>  
    	<filename>000000</filename>  
    	<source>  
    		<database>ILSVRC_2015</database>  
    	</source>  
    	<size>  
    		<width>1280</width>  
    		<height>720</height>  
    	</size>  
    	<object>  
    		<trackid>0</trackid>  
    		<name>n02691156</name>  
    		<bndbox>  
    			<xmax>659</xmax>  
    			<xmin>592</xmin>  
    			<ymax>375</ymax>  
    			<ymin>334</ymin>  
    		</bndbox>  
    		<occluded>0</occluded>  
    		<generated>0</generated>  
    	</object>  
    </annotation>  
      
  
---|---  
  
ElementTree生来就是为了处理XML, 它在Python标准库中有两种实现：一种是纯Python实现的, 如xml.etree.ElementTree, 另一种是速度快一点的xml.etree.cElementTree. 注意：尽量使用C语言实现的那种, 因为它速度更快, 而且消耗的内存更少.

  * a. 遍历根节点的下一层 
  * b. 下标访问各个标签、属性、文本
  * c. 查找root下的指定标签
  * d. 遍历XML文件
  * e. 修改XML文件
        
        1  
        2  
        3  
        4  
        5  
        

| 
        
        import  os, sys  
        try:  
            import xml.etree.cElementTree as ET  
        except:  
            import xml.etree.ElementTree as ET  
          
  
---|---  

## 解析xml文件
    
    
    1  
    2  
    3  
    4  
    5  
    6  
    7  
    8  
    

| 
    
    
    xmlFilePath = os.path.abspath('000000.xml')  
      
    try:  
        tree = ET.parse(xmlFilePath) # 或者 tree = ET.ElementTree(xmlFilePath)  
        root = tree.getroot() # 获取根节点  
    except Exception as e:  
        print('parse xml failed!')  
        sys.exit()  
      
  
---|---  
  
## 逐层遍历
    
    
    1  
    2  
    3  
    

| 
    
    
    print(root.tag, root.attrib, root.text)  
    for child in root:  
        print(child.tag, child.attrib, child.text)  
      
  
---|---  
  
## 递归遍历全部:
    
    
    1  
    2  
    3  
    4  
    5  
    6  
    7  
    

| 
    
    
    def traverseXml(element):  
        if len(element) > 0: # 叶节点的len为0  
            for child in element:  
                print(child.tag, child.attrib)  
                traverseXml(child)  
      
    traverseXml(root)  
      
  
---|---  
  
## 根据签名查找需要的标签
    
    
    1  
    2  
    3  
    4  
    5  
    

| 
    
    
    item_lists = root.findall('item') # 只能找到儿子, 不能找到孙子, 返回的是儿子们组成的列表  
    item = root.find('item') # 返回的是单个的儿子  
    print(root)  
    print(item_lists)  
    print(item)  
      
  
---|---  
  
## 获取叶子节点的值

## 当访问到叶子节点时, 就可以利用 text 来得到相应的标签了
    
    
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
    

| 
    
    
    obj_bbox_set =[]  
    objects = root.findall('object')  
    for obj in objects:  
        obj_name = obj.find('name').text  
        bbox = obj.find('bndbox')  
        x1 = int(bbox.find('xmin').text)  
        y1 = int(bbox.find('ymin').text)  
        x2 = int(bbox.find('xmax').text)  
        y2 = int(bbox.find('ymax').text)  
        obj_bbox_set.append([x1, x2, y1, y2, obj_name])  
    print(obj_bbox_set)  
      
  
---|---
