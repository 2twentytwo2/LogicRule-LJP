import json

# 读取 JSON 数据
def read_json_data(file_path):
    data = []
    with open(file_path, encoding="utf-8") as file:
        for idx, line in enumerate(file.readlines()):
            case = json.loads(line)
            data.append(case)
    print(f"{file_path} 一共有 {len(data)}条数据")
    return data

# 分析特定 class 标签对应的 address 标签种类
'''
分析特定 class 标签对应的 address 标签种类
'''
def analyze_addresses(data, target_class, target_class_name,next_name):
    relevant_articles_types = []

    if target_class_name =='accusation':
        a = []

    if target_class_name == 'relevant_articles' and next_name == 'accusation': # 法条对应的判决
        print('????')
        for item in data:
            item_meta = item['meta']
            if 'relevant_articles' in item_meta and item_meta['relevant_articles'] == target_class and 'accusation' in item_meta:
                # print(f"target_class={target_class}:  article={item_meta['relevant_articles']}")
                if item_meta['accusation'] not in relevant_articles_types:
                    relevant_articles_types.append(item_meta['accusation'])
    
    if target_class_name =='relevant_articles' and next_name =='imprisonment': # 判决对应的刑期
        for item in data:
            if 'meta' in item:
                item_meta = item['meta']
                if 'relevant_articles' in item_meta and item_meta['relevant_articles'] == target_class and 'imprisonment' in item_meta['term_of_imprisonment']:
                    # print(f"target_class={target_class}:  article={item_meta['relevant_articles']}")
                    if item_meta['term_of_imprisonment']['imprisonment'] not in relevant_articles_types:
                        relevant_articles_types.append(item_meta['term_of_imprisonment']['imprisonment'])

    if target_class_name =='accusation' and next_name =='imprisonment': 
        for item in data:
            if 'meta' in item:
                item_meta = item['meta']
                if 'accusation' in item_meta and item_meta['accusation'] == target_class and 'imprisonment' in item_meta['term_of_imprisonment']:
                    # print(f"target_class={target_class}:  article={item_meta['relevant_articles']}")
                    if item_meta['term_of_imprisonment']['imprisonment'] not in relevant_articles_types:
                        relevant_articles_types.append(item_meta['term_of_imprisonment']['imprisonment'])

    print(f"{target_class_name}={target_class}时{next_name}一共有{len(relevant_articles_types)}个类别")
    return relevant_articles_types 

def getClasslist(name,data):
    res = []
    for item in data:
        if 'meta' in item and name in item['meta']: # meta外面的类别
            # print(item['meta'][name])
            if item['meta'][name] not in res:
                res.append(item['meta'][name])
        # if 'meta' in item:  # meta里面的类别
        #     item_meta = item['meta']
        #     if item_meta['term_of_imprisonment']['imprisonment']  not in res:
        #         res.append(item_meta['term_of_imprisonment']['imprisonment'] )
    print(f"{name}类目一共有{len(res)}个类别")
    return res

# 主函数
def mainf(file_path):
    #file_path = 'your_json_file.json'  # 替换为实际的 JSON 文件路径
    classname = ['accusation','relevant_articles','imprisonment','punish_of_money']
    target_class_name = classname[0]
    next_name = classname[2]

    data = read_json_data(file_path)
    target_class_list = getClasslist(target_class_name,data)
    print(f" {target_class_name} 标签对应的种类有: {target_class_list}")
    for target_class in target_class_list:
        result = analyze_addresses(data, target_class, target_class_name,next_name)
        print(f"当 {target_class_name}  标签为 '{target_class}' 时，{next_name} 标签对应的种类有: {result}")



if __name__ == "__main__":
    cail18_path = '/data2/zhZH/PLJP/data/precedent_database/precedent_case.json'
    cjo22_path2 = '/data2/zhZH/PLJP/data/testset/cjo22/testset.json'
    mainf(cail18_path) 
