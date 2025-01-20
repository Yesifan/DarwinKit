# 草案

几个页面逻辑：

1. 页面加载的一开始，应该有2个部件就显示在画布中，一个是输入元件：供用户输入维度（目前可以支持1d (W)， 2d (C, W)， 3d(C,H,W)）三种；一个是输出元件，供用户确定输出结果的最终维度（他对应一个不带lif的linear层，输出维度由问题决定）；用户可操作的其实是2个元件中间的部分，也即不断的按顺序搭建神经网络结构

2. 因为现在只是最简单的按顺序从上到下搭建，每拖拉一个模块，需要有一个箭头自动生成表示前后关系

3. 在1d输入的情况，conv模块是不可以调用的，只能调用最简单的linear模块, 然后按顺序拖拉linear模块完成一个MLP的结构

4. 在2/3d输入情况下，可以将页面做的再死点？比如分成2部分，例如input->卷积模块部分->全连接部分->输出。

5. 在任何conv模块后按顺序接一个linear模块，意味着中间需要有个flatten操作自动添加在二者之间，只有逻辑4情况下会出现这种需求

   例如：

   ```python
   layer.Conv2d(12, 24),
   layer.flatten(),
   layer.Linear(100, 10)
   ```

6. Input 2d的情况是调用conv的时候自动设置为conv1d， input 3d则是设置conv2d，同理conv1d/2d接pool的维度对应也是1d/2d

7. linear后面不允许接pool，只允许接linear；

8. （可选）最好可以带一个输入输出维度的检验，尤其是当conv后调用一个linear时，linear的输入维度实际是可以按之前的搭建顺序进行计算得出的，这样能避免用户的失误？



算子：

卷积层 conv(in_channel, out_channel, kernel_size, stride, padding)

池化层（默认用的是maxpool,但是显示成pool就行） pool(kernel_size)

全连接层 Linear(in_dim, output_dim)

注：目前算子挺少的,感觉不需要一个box去让用户拖拽，似乎在一个画布模块内弄一个+号操作不断增加层数就够了



代码输出：

1. conv和linear的代码输出后面都应该带一个lif，例如：

   ```python
   Layer.linear(100, 10),
   Neuron.LIFNode(),
   ```

这样2行代表一个全连接层算子， 卷积层算子同理

2. 展示在页面的格式应当像一个普通的模型文件一样，包含包的引用，类的声明与初始化函数，以及forward函数，如：

   ```python
   import torch
   import torch.nn as nn
   from spikingjelly.activation_based import neuron, layer, functional
   
   class Model(nn.Module):
     def __init__(self, C=None, H=None, W=None): # 这里取决于input那里的维度
     	super().__init__()
       
       self.customized_layers = nn.Sequential(
         layer.conv2d(C, 12, 3, 1, 1),
         neuron.LIFNode(detach_reset=True),
         layer.maxpool2d(2)
         layer.flatten(),
         layer.linear(50, 30),
         neuron.LIFNode(detach_reset=True),
       )
       
       self.output_layer = layer.linear(30, 10)
       
       functional.set_step_mode('m')
       
      def forward(self, x):
       # make sure dimension of x is (T, C, H, W) C，H，w取决于input那里设置的维度，这个注释需要也自动生成，用来提示
       functional.reset_net(self)
       x = self.customized_layers(x)
       x = self.output_layer(x)
       return x
   ```

   



