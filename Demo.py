#这只是一个演示程序
import numpy
#S函数
import scipy.special
#显示图片
import matplotlib.pyplot 
# 从 PNG 图像文件加载数据的帮助程序
import imageio
# glob 有助于使用模式选择多个文件
import glob
#神经网络class类
class neuralNetwork :
# 初始化神经网络
    def __init__(self, inputnodes, hiddennodes, outputnodes,learningrate) :
# 在每个输入、隐藏、输出层中设置节点数
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
# 链接权重WIH和WHO
# 数组内的权重是w_i_j的，其中链接是从节点 I 到下一层的节点 J
# w11 w21
# w12 w22 etc
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5),
        (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5),
        (self.onodes, self.hnodes))
        # 学习率
        self.lr = learningrate
        # 激活函数是 S 函数
        self.activation_function = lambda x: scipy.special.expit(x)
        self.inverse_activation_function = lambda x: scipy.special.logit(x)
        pass
    # 训练神经网络
    def train(self, inputs_list, targets_list) :
# 将输入列表转换为二维数组
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        # 将信号计算到隐藏层
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 计算从隐藏层发出的信号
        hidden_outputs = self.activation_function(hidden_inputs)
        # 将信号计算到最终输出层
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 计算从最终输出层发出的信号
        final_outputs = self.activation_function(final_inputs)
        # 输出层误差为（目标 - 实际）
        output_errors = targets - final_outputs
        # 隐藏层误差是output_errors，按权重拆分，在隐藏节点处重新组合
        hidden_errors = numpy.dot(self.who.T, output_errors)
        # 更新隐藏图层和输出图层之间链接的权重
        self.who += self.lr * numpy.dot((output_errors *
        final_outputs * (1.0 - final_outputs)),
        numpy.transpose(hidden_outputs))
        # 更新输入图层和隐藏图层之间链接的权重
        self.wih += self.lr * numpy.dot((hidden_errors *
        hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        pass
        # 查询神经网络
    def query(self, inputs_list) :
        # 将输入列表转换为二维数组
        inputs = numpy.array(inputs_list, ndmin=2).T
        # 将信号计算到隐藏层
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 计算从隐藏层发出的信号
        hidden_outputs = self.activation_function(hidden_inputs)
        # 将信号计算到最终输出层
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 计算从最终输出层发出的信号
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
        # 输入、隐藏和输出节点数
    # 反向查询神经网络
    # 我们将对每个项目使用相同的术语，
    # 例如，目标是网络右侧的值，尽管用作输入
    # 例如 hidden_output 是中间节点右侧的信号
    def backquery(self, targets_list):
        # 将目标列表转置为垂直数组
        final_outputs = numpy.array(targets_list, ndmin=2).T
        
        # 将信号计算到最终输出层
        final_inputs = self.inverse_activation_function(final_outputs)

        # 计算隐藏层外的信号
        hidden_outputs = numpy.dot(self.who.T, final_inputs)
        # 将它们缩小到 0.01 到 .99
        hidden_outputs -= numpy.min(hidden_outputs)
        hidden_outputs /= numpy.max(hidden_outputs)
        hidden_outputs *= 0.98
        hidden_outputs += 0.01
        
        # 计算进入隐藏层的信号
        hidden_inputs = self.inverse_activation_function(hidden_outputs)
        
        # 计算输出输入层的信号
        inputs = numpy.dot(self.wih.T, hidden_inputs)
        # 将它们缩小到 0.01 到 .99
        inputs -= numpy.min(inputs)
        inputs /= numpy.max(inputs)
        inputs *= 0.98
        inputs += 0.01
        
        return inputs

input_nodes = 784
hidden_nodes = 200
output_nodes = 10
# 学习率
learning_rate = 0.1
# 创建神经网络实例
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)
# 将 mnist 训练数据 CSV 文件加载到列表中
training_data_file = open("D:/csv/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()
# 训练神经网络
# epochs 是训练数据集用于训练的次数
epochs = 5
for e in range(epochs):
    # 遍历训练数据集中的所有记录
    for record in training_data_list:
        # 将记录拆分为“，”逗号
        all_values = record.split(',')
# 缩放和移动输入
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
# 创建目标输出值（全部为 0.01，所需标签为 0.99 除外）
        targets = numpy.zeros(output_nodes) + 0.01
        # all_values[0] 是此记录的目标标签
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass
# 确保绘图在此笔记本内，而不是外部窗口

# load the mnist test data CSV file into a list
test_data_file = open("D:/csv/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()
# 测试神经网络

# 网络性能的记分卡，最初为空
scorecard = []

#遍历测试数据集中的所有记录
for record in test_data_list:
    # 将记录拆分为“，”逗号
    all_values = record.split(',')
    # 正确答案是第一值
    correct_label = int(all_values[0])
    # 缩放和移动输入
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # 查询网络
    outputs = n.query(inputs)
    # 最大值的索引对应于标签
    label = numpy.argmax(outputs)
    # 将正确或不正确附加到列表
    if (label == correct_label):
        # 网络的答案与正确答案匹配，在记分卡中添加 1
        scorecard.append(1)
    else:
        # 网络的答案与正确答案不匹配，将 0 添加到记分卡
        scorecard.append(0)
        pass
    
    pass
# 计算性能分数，正确答案的比例
scorecard_array = numpy.asarray(scorecard)
print ("performance = ", scorecard_array.sum() / scorecard_array.size)
# 向后运行网络，给定一个标签，看看它产生什么图像

# 要测试的标签
label = 0
# 为此标签创建输出信号
targets = numpy.zeros(output_nodes) + 0.01
# all_values[0] 是此记录的目标标签
targets[label] = 0.99
print(targets)

#获取图像数据
image_data = n.backquery(targets)

# 绘制图像数据
matplotlib.pyplot.imshow(image_data.reshape(28,28), cmap='Greys', interpolation='None')
matplotlib.pyplot.show()
#我们自己的图像测试数据集
# our own image test data set
our_own_dataset = []
# 将 PNG 图像数据加载为测试数据集
for image_file_name in glob.glob('number/number?.png'):
    
    # 使用文件名设置正确的标签
    label = int(image_file_name[-5:-4])
    
    # 将图像数据从 PNG 文件加载到数组中
    print ("loading ... ", image_file_name)
    img_array = imageio.imread(image_file_name, as_gray=True)
    
    # 从 28x28 重塑为 784 个值的列表，反转值
    img_data  = 255.0 - img_array.reshape(784)
    
    # 然后将数据缩放到 0.01 到 1.0 的范围
    img_data = (img_data / 255.0 * 0.99) + 0.01
    print(numpy.min(img_data))
    print(numpy.max(img_data))
    
    # 将标签和图像数据附加到测试数据集
    record = numpy.append(label,img_data)
    our_own_dataset.append(record)
    
    pass
# 用我们自己的图像测试神经网络

# 记录以进行测试
item = 6

# 绘图图像
matplotlib.pyplot.imshow(our_own_dataset[item][1:].reshape(28,28), cmap='Greys', interpolation='None')
matplotlib.pyplot.show()
# 正确答案是第一值
correct_label = our_own_dataset[item][0]
# 数据是剩余值
inputs = our_own_dataset[item][1:]

# 查询网络
outputs = n.query(inputs)
print (outputs)

# 最大值的索引对应于标签
label = numpy.argmax(outputs)
print("network says ", label)
# 将正确或不正确附加到列表
if (label == correct_label):
    print ("match!")
else:
    print ("no match!")
    pass
