from datasets import load_dataset

ds = load_dataset("shibing624/sharegpt_gpt4")

# print(ds['conversations'][0]['messages'])

# # 查看数据集的结构
# print(ds)

# # 查看数据集的字段
# print(ds['train'].column_names)

# # 查看数据集的大小
# print(len(ds['train']))

# print(len(ds['train'][0]['conversations']))

# 输出ds['train'][0]['conversations']每一个元素
for i in range(len(ds['train'][0]['conversations'])):
    print(ds['train'][0]['conversations'][i])


# 读取ds['train'][i]['conversations']的长度，将各个元素的长度存入一个列表
# lengths = []  
# for i in range(len(ds['train'])):
#     lengths.append(len(ds['train'][i]['conversations'])/2)
# # 输出列表
# print(lengths)