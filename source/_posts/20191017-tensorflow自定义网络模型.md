---
title: tensorflow自定义网络模型
categories: 深度学习
date: 2019-10-17 00:00:00
tags: 深度学习
  - python
  - tensorflow
---

## Slim  
  
TF-Slim 模块是 TensorFlow 中最好用的 API 之一。尤其是里面引入的 arg_scope、model_variables、repeat、stack。  
TF-Slim 是 TensorFlow 中一个用来构建、训练、评估复杂模型的轻量化库。TF-Slim 模块可以和 TensorFlow 中其它API混合使用。

### Slim模块的导入
    
    

```python
    import tensorflow.contrib.slim as slim
```

  
### Slim 构建模型

可以用 slim、variables、layers 和 scopes 来十分简洁地定义模型。下面对各个部分进行了详细描述：

#### Slim变量（Variables）
    
    

```
    weights = slim.variable('weights',
                            shape=[10, 10, 3 , 3],
                            initializer=tf.truncated_normal_initializer(stddev=0.1),
                            regularizer=slim.l2_regularizer(0.05),
                            device='/CPU:0')
    ~
```

  
#### Slim 层（Layers）

使用基础（plain）的 TensorFlow 代码：  

    
    

```
    input = ...
    with tf.name_scope('conv1_1') as scope:
      kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                               stddev=1e-1), name='weights')
      conv = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='SAME')
      biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                           trainable=True, name='biases')
      bias = tf.nn.bias_add(conv, biases)
      conv1 = tf.nn.relu(bias, name=scope)
```

  
为了避免代码的重复。Slim 提供了很多方便的神经网络 layers 的高层 op。例如：与上面的代码对应的 Slim 版的代码：  

    
    

```
    input = ...
    net = slim.conv2d(input, 128, [3, 3], scope='conv1_1')
```

  
### slim.arg_scope（） 函数的使用

这个函数的作用是给list_ops中的内容设置默认值。但是每个list_ops中的每个成员需要用@add_arg_scope修饰才行。所以使用slim.arg_scope（）有两个步骤：

  * 使用@slim.add_arg_scope修饰目标函数
  * 用 slim.arg_scope（）为目标函数设置默认参数.  
例如如下代码；首先用@slim.add_arg_scope修饰目标函数fun1（），然后利用slim.arg_scope（）为它设置默认参数。
        

```python
```python
import tensorflow as tf
slim =tf.contrib.slim
@slim.add_arg_scope
def fun1(a=0,b=0):
return (a+b)
with slim.arg_scope([fun1],a=10):
x=fun1(b=30)
print(x)
```
运行结果:40
参考链接：
<https://blog.csdn.net/u013921430/article/details/80915696>
### 其他用法见参考链接
<https://blog.csdn.net/wanttifa/article/details/90208398>
## 查看ckpt中变量的几种方法
查看ckpt中变量的方法有三种：
  * 在有model的情况下，使用tf.train.Saver进行restore
  * 使用tf.train.NewCheckpointReader直接读取ckpt文件，这种方法不需要model。
  * 使用tools里的freeze_graph来读取ckpt
Tips:
  * 如果模型保存为.ckpt的文件，则使用该文件就可以查看.ckpt文件里的变量。ckpt路径为 model.ckpt
  * 如果模型保存为.ckpt-xxx-data (图结构)、.ckpt-xxx.index (参数名)、.ckpt-xxx-meta (参数值)文件，则需要同时拥有这三个文件才行。并且ckpt的路径为 model.ckpt-xxx
### 1.基于model来读取ckpt文件里的变量
1.首先建立起model
2.从ckpt中恢复变量
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
        with tf.Graph().as_default() as g:
          #建立model
          images, labels = cifar10.inputs(eval_data=eval_data)
          logits = cifar10.inference(images)
          top_k_op = tf.nn.in_top_k(logits, labels, 1)
          #从ckpt中恢复变量
          sess = tf.Session()
          saver = tf.train.Saver() #saver = tf.train.Saver(...variables...) # 恢复部分变量时，只需要在Saver里指定要恢复的变量
          save_path = 'ckpt的路径'
          saver.restore(sess, save_path) # 从ckpt中恢复变量
```


注意：基于model来读取ckpt中变量时，model和ckpt必须匹配。

### 2.使用tf.train.NewCheckpointReader直接读取ckpt文件里的变量，使用tools.inspect_checkpoint里的print_tensors_in_checkpoint_file函数打印ckpt里的东西
    
    

```python
    #使用NewCheckpointReader来读取ckpt里的变量
    from tensorflow.python import pywrap_tensorflow
    checkpoint_path = os.path.join(model_dir, "model.ckpt")
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path) #tf.train.NewCheckpointReader
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
      print("tensor_name: ", key)
      #print(reader.get_tensor(key))
    #使用print_tensors_in_checkpoint_file打印ckpt里的内容
    from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
    print_tensors_in_checkpoint_file(file_name, #ckpt文件名字
                     tensor_name, # 如果为None,则默认为ckpt里的所有变量
                     all_tensors, # bool 是否打印所有的tensor，这里打印出的是tensor的值，一般不推荐这里设置为False
                     all_tensor_names) # bool 是否打印所有的tensor的name
    #上面的打印ckpt的内部使用的是pywrap_tensorflow.NewCheckpointReader所以要掌握NewCheckpointReader
```

  
### 3.使用tools里的freeze_graph来读取ckpt
    
    

```python
    from tensorflow.python.tools import freeze_graph
    freeze_graph(input_graph, #=some_graph_def.pb
           input_saver,
           input_binary,
           input_checkpoint, #=model.ckpt
           output_node_names, #=softmax
           restore_op_name,
           filename_tensor_name,
           output_graph, #='./tmp/frozen_graph.pb'
           clear_devices,
           initializer_nodes,
           variable_names_whitelist='',
           variable_names_blacklist='',
           input_meta_graph=None,
           input_saved_model_dir=None,
           saved_model_tags='serve',
           checkpoint_version=2)
    #freeze_graph_test.py讲述了怎么使用freeze_grapg。
```

  
参考链接：  
<https://www.jb51.net/article/142183.htm>

## control_dependencies

tf.control_dependencies(control_inputs)  
Wrapper for Graph.control_dependencies() using the default graph.  
See Graph.control_dependencies() for more details.  
此函数指定某些操作执行的依赖关系  
返回一个控制依赖的上下文管理器，使用 with 关键字可以让在这个上下文环境中的操作都在 control_inputs 执行  

    
    

```
    1 with tf.control_dependencies([a, b]):
    2     c = ....
    3     d = ...
```

  
在执行完 a，b 操作之后，才能执行 c，d 操作。意思就是 c，d 操作依赖 a，b 操作  

    
    

```
    1 with tf.control_dependencies([train_step, variable_averages_op]):
    2     train_op = tf.no_op(name='train')
```

  
tf.no_op()表示执行完 train_step, variable_averages_op 操作之后什么都不做  
参考链接：  
<http://www.tensorfly.cn/tfdoc/api_docs/python/framework.html#Graph.control_dependencies>

## TensorBoard

### 在TensorBoard中可视化图形

构建您的网络，创建一个会话(session)，然后创建一个TensorFlow File Writer对象  
File Writer定义存储TensorBoard文件的路径，以及TensorFlow graph对象sess.graph是第二个参数。  

    
    

```
    writer = tf.summary.FileWriter(STORE_PATH, sess.graph)
```

  
当创建一个TensorFlow网络后，定义并运行File Writer时，就可以启动TensorBoard来可视化图形。要定义File Writer并将图形发送给它，运行以下命令:  

    
    

```
    # start the session
    with tf.Session() as sess:
    writer = tf.summary.FileWriter(STORE_PATH, sess.graph)
```

  
### 启动TensorBoard
    
    

```
    tensorboard --logdir=STORE_PATH
```

  
### 名称空间（Namespaces）

名称空间是一种作用域，可以用它来包围图形组件，以便将它们组合在一起。通过这样的操作，名称空间中的细节将被折叠成TensorBoard计算图形可视化中的单个名称空间节点。要在TensorFlow中创建名称空间，可以使用Python with功能，如下所示：  

    
    

```
    with tf.name_scope("layer_1"):
    # now declare the weights connecting the input to the hidden  layer
     W1 = tf.Variable(tf.random_normal([784, 300], stddev=0.01), name='W')
             b1 = tf.Variable(tf.random_normal([300]), name='b')
             hidden_logits = tf.add(tf.matmul(x_sc, W1), b1)
             hidden_out = tf.nn.sigmoid(hidden_logits)
```

  
还可以使用tf.variable_scope()代替tf.name_scope()。变量作用域是TensorFlow中的get_variable()变量共享机制的一部分。

### 标量总结（Scalar summaries）

在网络中的任何位置，都可以记录标量(即单个实值)数量，以便在TensorBoard中显示。这对于跟踪诸如训练准确率的提高或损失函数的减少，或研究分布的标准差等方面都很有用。执行起来很容易。例如，下面的代码展示了如何在这个图中记录accuracy标量:  

    
    

```
    # add a summary to store the
    accuracytf.summary.scalar('acc_summary', accuracy)
```

  
第一个参数是要在TensorBoard可视化中给出标量的名称，第二个参数是要记录的操作(必须返回一个实值)。scalar()调用的输出是一个操作。在上面的代码中，我没有将这个操作分配给Python中的任何变量，但是如果用户愿意，可以这样做。然而，与TensorFlow中的其他操作一样，这些汇总操作在运行之前不会执行任何操作。根据开发人员想要观察的内容，在任何给定的图中通常都会运行许多可视化函数，因此有一个方便的助手函数merge_all()。这将把图中的所有函数调用合并在一起，这样您只需调用merge操作，它将为您收集所有其他函数操作并记录数据。它是这样的:  

    
    

```
    merged = tf.summary.merge_all()
```

  
### 图像可视化
    
    

```
    # add summary
    if reuse_variables is None:
        tf.summary.image('input', images)
        tf.summary.image('score_map', score_maps)
        tf.summary.image('score_map_pred', f_score * 255)
        tf.summary.image('geo_map_0', geo_maps[:, :, :, 0:1])
        tf.summary.image('geo_map_0_pred', f_geometry[:, :, :, 0:1])
        tf.summary.image('training_masks', training_masks)
        tf.summary.scalar('model_loss', model_loss)
        tf.summary.scalar('total_loss', total_loss)
```

  
## 文本检测模型EAST的搭建

### 数据加载
    
    

```python
    def load_annoataion(p):
        '''
        load annotation from the text file
        :param p:
        :return:
        '''
        text_polys = []
        text_tags = []
        if not os.path.exists(p):
            return np.array(text_polys, dtype=np.float32)
        with open(p, 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                label = line[-1]
                # strip BOM. \ufeff for python3,  \xef\xbb\bf for python2
                line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]
                x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
                text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
                if label == '*' or label == '###':
                    text_tags.append(True)
                else:
                    text_tags.append(False)
            return np.array(text_polys, dtype=np.float32), np.array(text_tags, dtype=np.bool)
    def generator(input_size=512, batch_size=32,
                background_ratio=3./8,
                random_scale=np.array([0.5, 1, 2.0, 3.0]),
                vis=False):
        image_list = np.array(get_images())
        print('{} training images in {}'.format(
            image_list.shape[0], FLAGS.training_data_path))
        index = np.arange(0, image_list.shape[0])
        while True:
            np.random.shuffle(index)
            images = []
            image_fns = []
            score_maps = []
            geo_maps = []
            training_masks = []
            for i in index:
                try:
                    im_fn = image_list[i]
                    im = cv2.imread(im_fn)
                    # print im_fn
                    h, w, _ = im.shape
                    txt_fn = im_fn.replace(os.path.basename(im_fn).split('.')[1], 'txt')
                    if not os.path.exists(txt_fn):
                        print('text file {} does not exists'.format(txt_fn))
                        continue
                    text_polys, text_tags = load_annoataion(txt_fn)
                    text_polys, text_tags = check_and_validate_polys(text_polys, text_tags, (h, w))
                    # if text_polys.shape[0] == 0:
                    #     continue
                    # random scale this image
                    rd_scale = np.random.choice(random_scale)
                    im = cv2.resize(im, dsize=None, fx=rd_scale, fy=rd_scale)
                    text_polys *= rd_scale
                    # print rd_scale
                    # random crop a area from image
                    if np.random.rand() < background_ratio:
                        # crop background
                        im, text_polys, text_tags = crop_area(im, text_polys, text_tags, crop_background=True)
                        if text_polys.shape[0] > 0:
                            # cannot find background
                            continue
                        # pad and resize image
                        new_h, new_w, _ = im.shape
                        max_h_w_i = np.max([new_h, new_w, input_size])
                        im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
                        im_padded[:new_h, :new_w, :] = im.copy()
                        im = cv2.resize(im_padded, dsize=(input_size, input_size))
                        score_map = np.zeros((input_size, input_size), dtype=np.uint8)
                        geo_map_channels = 5 if FLAGS.geometry == 'RBOX' else 8
                        geo_map = np.zeros((input_size, input_size, geo_map_channels), dtype=np.float32)
                        training_mask = np.ones((input_size, input_size), dtype=np.uint8)
                    else:
                        im, text_polys, text_tags = crop_area(im, text_polys, text_tags, crop_background=False)
                        if text_polys.shape[0] == 0:
                            continue
                        h, w, _ = im.shape
                        # pad the image to the training input size or the longer side of image
                        new_h, new_w, _ = im.shape
                        max_h_w_i = np.max([new_h, new_w, input_size])
                        im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
                        im_padded[:new_h, :new_w, :] = im.copy()
                        im = im_padded
                        # resize the image to input size
                        new_h, new_w, _ = im.shape
                        resize_h = input_size
                        resize_w = input_size
                        im = cv2.resize(im, dsize=(resize_w, resize_h))
                        resize_ratio_3_x = resize_w/float(new_w)
                        resize_ratio_3_y = resize_h/float(new_h)
                        text_polys[:, :, 0] *= resize_ratio_3_x
                        text_polys[:, :, 1] *= resize_ratio_3_y
                        new_h, new_w, _ = im.shape
                        score_map, geo_map, training_mask = generate_rbox((new_h, new_w), text_polys, text_tags)
                    if vis:
                        fig, axs = plt.subplots(3, 2, figsize=(20, 30))
                        # axs[0].imshow(im[:, :, ::-1])
                        # axs[0].set_xticks([])
                        # axs[0].set_yticks([])
                        # for poly in text_polys:
                        #     poly_h = min(abs(poly[3, 1] - poly[0, 1]), abs(poly[2, 1] - poly[1, 1]))
                        #     poly_w = min(abs(poly[1, 0] - poly[0, 0]), abs(poly[2, 0] - poly[3, 0]))
                        #     axs[0].add_artist(Patches.Polygon(
                        #         poly * 4, facecolor='none', edgecolor='green', linewidth=2, linestyle='-', fill=True))
                        #     axs[0].text(poly[0, 0] * 4, poly[0, 1] * 4, '{:.0f}-{:.0f}'.format(poly_h * 4, poly_w * 4),
                        #                    color='purple')
                        # axs[1].imshow(score_map)
                        # axs[1].set_xticks([])
                        # axs[1].set_yticks([])
                        axs[0, 0].imshow(im[:, :, ::-1])
                        axs[0, 0].set_xticks([])
                        axs[0, 0].set_yticks([])
                        for poly in text_polys:
                            poly_h = min(abs(poly[3, 1] - poly[0, 1]), abs(poly[2, 1] - poly[1, 1]))
                            poly_w = min(abs(poly[1, 0] - poly[0, 0]), abs(poly[2, 0] - poly[3, 0]))
                            axs[0, 0].add_artist(Patches.Polygon(
                                poly, facecolor='none', edgecolor='green', linewidth=2, linestyle='-', fill=True))
                            axs[0, 0].text(poly[0, 0], poly[0, 1], '{:.0f}-{:.0f}'.format(poly_h, poly_w), color='purple')
                        axs[0, 1].imshow(score_map[::, ::])
                        axs[0, 1].set_xticks([])
                        axs[0, 1].set_yticks([])
                        axs[1, 0].imshow(geo_map[::, ::, 0])
                        axs[1, 0].set_xticks([])
                        axs[1, 0].set_yticks([])
                        axs[1, 1].imshow(geo_map[::, ::, 1])
                        axs[1, 1].set_xticks([])
                        axs[1, 1].set_yticks([])
                        axs[2, 0].imshow(geo_map[::, ::, 2])
                        axs[2, 0].set_xticks([])
                        axs[2, 0].set_yticks([])
                        axs[2, 1].imshow(training_mask[::, ::])
                        axs[2, 1].set_xticks([])
                        axs[2, 1].set_yticks([])
                        plt.tight_layout()
                        plt.show()
                        plt.close()
                    images.append(im[:, :, ::-1].astype(np.float32))
                    image_fns.append(im_fn)
                    score_maps.append(score_map[::4, ::4, np.newaxis].astype(np.float32))
                    geo_maps.append(geo_map[::4, ::4, :].astype(np.float32))
                    training_masks.append(training_mask[::4, ::4, np.newaxis].astype(np.float32))
                    if len(images) == batch_size:
                        yield images, image_fns, score_maps, geo_maps, training_masks
                        images = []
                        image_fns = []
                        score_maps = []
                        geo_maps = []
                        training_masks = []
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    continue
```

  
### 网络模型的搭建
    
    

```python
    def model(images, weight_decay=1e-5, is_training=True):
        '''
        define the model, we use slim's implemention of resnet
        '''
        images = mean_image_subtraction(images)
        with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
            logits, end_points = resnet_v1.resnet_v1_50(images, is_training=is_training, scope='resnet_v1_50')
        with tf.variable_scope('feature_fusion', values=[end_points.values]):
            batch_norm_params = {
            'decay': 0.997,
            'epsilon': 1e-5,
            'scale': True,
            'is_training': is_training
            }
            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.relu,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                weights_regularizer=slim.l2_regularizer(weight_decay)):
                f = [end_points['pool5'], end_points['pool4'],
                     end_points['pool3'], end_points['pool2']]
                for i in range(4):
                    print('Shape of f_{} {}'.format(i, f[i].shape))
                g = [None, None, None, None]
                h = [None, None, None, None]
                num_outputs = [None, 128, 64, 32]
                for i in range(4):
                    if i == 0:
                        h[i] = f[i]
                    else:
                        c1_1 = slim.conv2d(tf.concat([g[i-1], f[i]], axis=-1), num_outputs[i], 1)
                        h[i] = slim.conv2d(c1_1, num_outputs[i], 3)
                    if i <= 2:
                        g[i] = unpool(h[i])
                    else:
                        g[i] = slim.conv2d(h[i], num_outputs[i], 3)
                    print('Shape of h_{} {}, g_{} {}'.format(i, h[i].shape, i, g[i].shape))
                # here we use a slightly different way for regression part,
                # we first use a sigmoid to limit the regression range, and also
                # this is do with the angle map
                F_score = slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
                # 4 channel of axis aligned bbox and 1 channel rotation angle
                geo_map = slim.conv2d(g[3], 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * FLAGS.text_scale
                angle_map = (slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) - 0.5) * np.pi/2 # angle is between [-45, 45]
                F_geometry = tf.concat([geo_map, angle_map], axis=-1)
        return F_score, F_geometry
```

  
### loss函数的设计
    
    

```python
    def loss(y_true_cls, y_pred_cls,
             y_true_geo, y_pred_geo,
             training_mask):
        '''
        define the loss used for training, contraning two part,
        the first part we use dice loss instead of weighted logloss,
        the second part is the iou loss defined in the paper
        :param y_true_cls: ground truth of text
        :param y_pred_cls: prediction os text
        :param y_true_geo: ground truth of geometry
        :param y_pred_geo: prediction of geometry
        :param training_mask: mask used in training, to ignore some text annotated by ###
        :return:
        '''
        classification_loss = dice_coefficient(y_true_cls, y_pred_cls, training_mask)
        # scale classification loss to match the iou loss part
        classification_loss *= 0.01
        # d1 -> top, d2->right, d3->bottom, d4->left
        d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(value=y_true_geo, num_or_size_splits=5, axis=3)
        d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(value=y_pred_geo, num_or_size_splits=5, axis=3)
        area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
        area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
        w_union = tf.minimum(d2_gt, d2_pred) + tf.minimum(d4_gt, d4_pred)
        h_union = tf.minimum(d1_gt, d1_pred) + tf.minimum(d3_gt, d3_pred)
        area_intersect = w_union * h_union
        area_union = area_gt + area_pred - area_intersect
        L_AABB = -tf.log((area_intersect + 1.0)/(area_union + 1.0))
        L_theta = 1 - tf.cos(theta_pred - theta_gt)
        tf.summary.scalar('geometry_AABB', tf.reduce_mean(L_AABB * y_true_cls * training_mask))
        tf.summary.scalar('geometry_theta', tf.reduce_mean(L_theta * y_true_cls * training_mask))
        L_g = L_AABB + 20 * L_theta
        return tf.reduce_mean(L_g * y_true_cls * training_mask) + classification_loss
```

  
### train
    
    

```python
    def main(argv=None):
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
        config = None
        config.batch_size = FLAGS.batch_size_per_gpu * FLAGS.num_gpus
        if not tf.gfile.Exists(FLAGS.checkpoint_path):
            tf.gfile.MkDir(FLAGS.checkpoint_path)
        else:
            if not FLAGS.restore:
                tf.gfile.DeleteRecursively(FLAGS.checkpoint_path)
                tf.gfile.MkDir(FLAGS.checkpoint_path)
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        input_score_maps = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input_score_maps')
        if FLAGS.geometry == 'RBOX':
            input_geo_maps = tf.placeholder(tf.float32, shape=[None, None, None, 5], name='input_geo_maps')
        else:
            input_geo_maps = tf.placeholder(tf.float32, shape=[None, None, None, 8], name='input_geo_maps')
        input_training_masks = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input_training_masks')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps=10000, decay_rate=0.94, staircase=True)
        # add summary
        tf.summary.scalar('learning_rate', learning_rate)
        opt = tf.train.AdamOptimizer(learning_rate)
        # opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
        # split
        input_images_split = tf.split(input_images, len(gpus))
        input_score_maps_split = tf.split(input_score_maps, len(gpus))
        input_geo_maps_split = tf.split(input_geo_maps, len(gpus))
        input_training_masks_split = tf.split(input_training_masks, len(gpus))
        tower_grads = []
        reuse_variables = None
        for i, gpu_id in enumerate(gpus):
            with tf.device('/gpu:%d' % gpu_id):
                with tf.name_scope('model_%d' % gpu_id) as scope:
                    iis = input_images_split[i]
                    isms = input_score_maps_split[i]
                    igms = input_geo_maps_split[i]
                    itms = input_training_masks_split[i]
                    total_loss, model_loss = tower_loss(iis, isms, igms, itms, reuse_variables)
                    batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope))
                    reuse_variables = True
                    grads = opt.compute_gradients(total_loss)
                    tower_grads.append(grads)
        grads = average_gradients(tower_grads)
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        summary_op = tf.summary.merge_all()
        # save moving average
        variable_averages = tf.train.ExponentialMovingAverage(
            FLAGS.moving_average_decay, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
        # batch norm updates
        with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
            train_op = tf.no_op(name='train_op')
        saver = tf.train.Saver(tf.global_variables())
        summary_writer = tf.summary.FileWriter(FLAGS.checkpoint_path, tf.get_default_graph())
        init = tf.global_variables_initializer()
        if FLAGS.pretrained_model_path is not None:
            variable_restore_op = slim.assign_from_checkpoint_fn(FLAGS.pretrained_model_path, slim.get_trainable_variables(),
                                                                ignore_missing_vars=True)
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            if FLAGS.restore:
                print('continue training from previous checkpoint')
                ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
                saver.restore(sess, ckpt)
            else:
                sess.run(init)
                if FLAGS.pretrained_model_path is not None:
                    variable_restore_op(sess)
            # data_generator = icdar.get_batch(num_workers=FLAGS.num_readers,
            #                                 input_size=FLAGS.input_size,
            #                                 batch_size=FLAGS.batch_size_per_gpu * len(gpus))
            train_data_generator = icdar_single.get_batch_seq(num_workers=FLAGS.num_readers, config=config, is_training=True)
            start = time.time()
            for step in range(FLAGS.max_steps):
                data = next(train_data_generator)
                ml, tl, _ = sess.run([model_loss, total_loss, train_op], feed_dict={input_images: data[0],
                                                                                    input_score_maps: data[2],
                                                                                    input_geo_maps: data[3],
                                                                                    input_training_masks: data[4]})
                if np.isnan(tl):
                    print('Loss diverged, stop training')
                    break
                if step % 10 == 0:
                    avg_time_per_step = (time.time() - start)/10
                    avg_examples_per_second = (10 * FLAGS.batch_size_per_gpu * len(gpus))/(time.time() - start)
                    start = time.time()
                    print('Step {:06d}, model loss {:.4f}, total loss {:.4f}, {:.2f} seconds/step, {:.2f} examples/second'.format(
                        step, ml, tl, avg_time_per_step, avg_examples_per_second))
                if step % FLAGS.save_checkpoint_steps == 0:
                    saver.save(sess, FLAGS.checkpoint_path + 'model.ckpt', global_step=global_step)
                if step % FLAGS.save_summary_steps == 0:
                    _, tl, summary_str = sess.run([train_op, total_loss, summary_op], feed_dict={input_images: data[0],
                                                                                                input_score_maps: data[2],
                                                                                                input_geo_maps: data[3],
                                                                                                input_training_masks: data[4]})
                    summary_writer.add_summary(summary_str, global_step=step)
```

  
参考链接：  
<https://github.com/argman/EAST>
