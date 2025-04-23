# Motivation - 算法部分

* 参考[SpargeAttn](https://github.com/thu-ml/SpargeAttn)，提取了这篇文章中的前半部分（Step 1.  Sparse Block Online Prediction），获取到Block Sparse Mask后的部分都是自己写的。**不保证没bug**，但可以保证的是魔改原始llama的attn机制并且保存kv cache是可以走通的。
* 启动方式：所需的虚拟环境就是transformers、torch这些比较基本的，安装好虚拟环境后将llama部分的两个文件放入 tranformers库中，路径为`transformers/models/llama/`这个文件夹中。
  * 运行`cal_llama_ori_kv_cache.py`可以获取kv cach。**运行前要在modeling_llama.py文件中修改是否添加稀疏化的逻辑**。初期先通过这种方式控制，后期可以考虑在config里设置。
  * 运行`cal_kv_cache_sim.py`可计算相似度，方式为每层的kv cache取平均。
* 注意事项：
  * LLaMA3.1-8B原生支持的attn implementation是spda attention。本实验中方便处理统一采用eager实现方式。
  * 不论是原模型还是稀疏化后的模型，都会出现**复读**的现象。原模型即使用spda也复读，不知道是不是模型自身原因。
* 运行案例：
  * query
    ```
    query = """
    A robe takes 2 bolts of blue fiber and half that much white fiber.
    How many bolts in total does it take?
    """
    ```
  * 得出：
    * K的余弦相似度为：0.9787
    * V的余弦相似度为：0.9249
  * 该结果在更长文本的情况下有待进一步验证。