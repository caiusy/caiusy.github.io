---
title: LNMS
date: 2019-08-16 00:00:00
categories:
  - 深度学习
tags:
  - OpenCV
---

## locality NMS  
  
LNMS是在EAST文本检测中提出的．主要原因：文本检测面临的是成千上万个几何体，如果用普通的NMS，其计算复杂度，n是几何体的个数，这是不可接受的．对上述时间复杂度问题，EAST提出了基于行合并几何体的方法，当然这是基于邻近几个几何体是高度相关的假设．注意：这里合并的四边形坐标是通过两个给定四边形的得分进行加权平均的，也就是说这里是“平均”而不是”选择”几何体*,目的是减少计算量．  
基本步骤  
1.先对所有的output box集合结合相应的阈值（大于阈值则进行合并，小于阈值则不和并），依次遍历进行加权合并，得到合并后的bbox集合；  
2.对合并后的bbox集合进行标准的NMS操作  

    
    
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
    

| 
    
    
    def detect(score_map, geo_map, timer, score_map_thresh=1e-5, box_thresh=1e-8, nms_thres=0.1):  
        '''  
        restore text boxes from score map and geo map  
        :param score_map: bs* 128 * 128 * 1  
        :param geo_map: ## geo_map = bs * 128 * 128 * 5  
        :param timer:  
        :param score_map_thresh: threshhold for score map  
        :param box_thresh: threshhold for boxes  
        :param nms_thres: threshold for nms  
        :return:  
        '''  
        if len(score_map.shape) == 4:  
            score_map = score_map[0, :, :, 0]  
            geo_map = geo_map[0, :, :, ]  
        # filter the score map  
        xy_text = np.argwhere(score_map > score_map_thresh)  
        # sort the text boxes via the y axis  
        xy_text = xy_text[np.argsort(xy_text[:, 0])]  
        # restore  
        start = time.time()  
        text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2  
        #print('{} text boxes before nms'.format(text_box_restored.shape[0]))  
        boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)  
        boxes[:, :8] = text_box_restored.reshape((-1, 8))  
        boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]  
        timer['restore'] = time.time() - start  
      
        # 得到box 的坐标以及分数  
          
        # nms part  
        start = time.time()  
        # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)  
        boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)  
        timer['nms'] = time.time() - start  
        if boxes.shape[0] == 0:  
            return None, timer  
      
        # here we filter some low score boxes by the average score map, this is different from the orginal paper  
        for i, box in enumerate(boxes):  
            mask = np.zeros_like(score_map, dtype=np.uint8)  
            cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)  
            boxes[i, 8] = cv2.mean(score_map, mask)[0]  
        boxes = boxes[boxes[:, 8] > box_thresh]  
        return boxes, timer  
      
  
---|---  
  
* * *
    
    
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
    

| 
    
    
    import numpy as np  
    from shapely.geometry import Polygon  
      
    def intersection(g, p):  
        #取g,p中的几何体信息组成多边形  
        g = Polygon(g[:8].reshape((4, 2)))  
        p = Polygon(p[:8].reshape((4, 2)))  
      
        # 判断g,p是否为有效的多边形几何体  
        if not g.is_valid or not p.is_valid:  
            return 0  
      
        # 取两个几何体的交集和并集  
        inter = Polygon(g).intersection(Polygon(p)).area  
        union = g.area + p.area - inter  
        if union == 0:  
            return 0  
        else:  
            return inter/union  
      
    def weighted_merge(g, p):  
        # 取g,p两个几何体的加权（权重根据对应的检测得分计算得到）  
        g[:8] = (g[8] * g[:8] + p[8] * p[:8])/(g[8] + p[8])  
      
        #合并后的几何体的得分为两个几何体得分的总和  
        g[8] = (g[8] + p[8])  
        return g  
      
    def standard_nms(S, thres):  
        #标准NMS  
        order = np.argsort(S[:, 8])[::-1]  
        keep = []  
        while order.size > 0:  
            i = order[0]  
            keep.append(i)  
            ovr = np.array([intersection(S[i], S[t]) for t in order[1:]])  
            inds = np.where(ovr <= thres)[0]  
            order = order[inds+1]  
      
        return S[keep]  
      
    def nms_locality(polys, thres=0.3):  
        '''  
        locality aware nms of EAST  
        :param polys: a N*9 numpy array. first 8 coordinates, then prob  
        :return: boxes after nms  
        '''  
        S = []    #合并后的几何体集合  
        p = None   #合并后的几何体  
        for g in polys:  
            if p is not None and intersection(g, p) > thres:    #若两个几何体的相交面积大于指定的阈值，则进行合并  
                p = weighted_merge(g, p)  
            else:    #反之，则保留当前的几何体  
                if p is not None:  
                    S.append(p)  
                p = g  
        if p is not None:  
            S.append(p)  
        if len(S) == 0:  
            return np.array([])  
        return standard_nms(np.array(S), thres)  
      
    if __name__ == '__main__':  
        # 343,350,448,135,474,143,369,359  
        print(Polygon(np.array([[343, 350], [448, 135],  
                                [474, 143], [369, 359]])).area)  
      
  
---|---  
  
参考博客： <https://www.jianshu.com/p/4934875f7eb6>
