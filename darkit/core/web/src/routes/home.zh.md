<p>DarwinKit 是一个基于 PyTorch 的自然语言处理工具包，提供了一些常用的 NLP 模型和工具。</p>

---

<div>
  <a href="/docs" class="border-primary mr-1 border-r pr-2">文档</a>
  <a href="/downloads/packages/" data-no-translate>安装包</a>
</div>

---

DarwinKit 是一个主要面向初学者的自然语言处理工具包，提供了开箱即用的 NLP
模型和工具，并且提供了网页工具，方便用户快速上手。也面向研究者提供了编程接口，以编程的方式修改和运行
NLP 模型。

要特性是：

- 面向初学者的优势
  - 易于使用，且一次安装即可使用不同模型。
  - 提供了网页工具来训练和使用模型。并且可以组合不同的 数据集 - 标记化器 - 模型 来进行训练。
  - 提供可视化的方式查看训练损失等数据。
- 面向开发者的优势
  - 提供内置的数据集预处理方法。
  - 方便的与不同的数据集和标记化器集成。
  - 自动保存模型，无需手动保存模型，避免因为忘记保存模型而导致的模型丢失。
  - 保存模型权重的同时会保存模型的配置信息，避免因为模型配置信息丢失而导致的模型无法使用。以及方便不同配置的训练效果的对比。
  - 自动评估模型，无需手动设置评估模型的循环。
  - 提供内置的 log 工具，将训练期间的数据记录到文件中，方便后续查看。
  - 提供可视化的方式查看训练损失等数据。
    可以查看训练中和训练完成的模型的数据，对比不同模型的参数和训练效果，实时查看训练过程中的数据。