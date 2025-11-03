'''
    从原数据中提取出对应的案件
    格式和2月份的测试集一样
'''
import os
import sys
import json


# path = '/data2/zhZH/ljp-llm/mydata/analysis_data/pn-10-18-18-54_flat.jsonl'
# name = 'theft'


# path = '/data2/zhZH/ljp-llm/mydata/analysis_data/pnHurt-10-18-21-01_flat.jsonl'
# name = 'Hurt'

# path = '/data2/zhZH/ljp-llm/mydata/analysis_data/pn-Intentional Homicide-10-18-21-25_flat.jsonl'
# name = 'Intentional Homicide'

# path = '/data2/zhZH/ljp-llm/mydata/analysis_data/pn-lesuo-10-18-22-06_flat.jsonl'
# name = 'lesuo'


path = '/data2/zhZH/ljp-llm/mydata/analysis_data/pn-fire-10-18-22-15_flat.jsonl'
name = 'fire'

origin_path = '/data2/zhZH/ljp-llm/mydata/CAIL2018/data/BertCnn_test_top5.json'
write_path = '/data2/zhZH/ljp-llm/mydata/analysis_confuse'

def readjson_4list(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            stripped_line = line.strip() # 跳过空行
            if not stripped_line:
                continue
            try:
                data.append(json.loads(stripped_line))
            except json.JSONDecodeError as e:
                print(f"解析行出错: {e}\n问题行内容: {line[:100]}...")
    return data



def readjsonlist(path):
    try:
        # 打开并读取 JSON 文件
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)  # 直接解析文件对象
        
    except FileNotFoundError:
        print(f"错误：文件 '{path}' 不存在")
    except json.JSONDecodeError as e:
        print(f"错误：JSON 格式无效 - {e}")
    except Exception as e:
        print(f"发生未知错误：{e}")
    return data
    

def find_origindata(targetfact,datalist):
    
    for item in datalist:
        current_fact = item['fact']
        if targetfact == current_fact:
            return item
    
    return None

def writejsonlist(name,path,data):
    
    """
    将数据写入JSON文件
    参数:
        data (list): 要写入的列表数据
        file_path (str): JSON文件路径
    """
    file_path = f'{path}/{name}_confusedata.json'
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)  # 写入JSON文件内容  
        print(f"数据已成功写入 {file_path}")
    except Exception as e:
        print(f"写入文件 {file_path} 出错: {str(e)}")




testdatalist = readjson_4list(path)
origindatalist = readjsonlist(origin_path)
test_origin_list = []
for testitem in testdatalist:
    targetfact = testitem['fact']
    originitem = find_origindata(targetfact,origindatalist)
    test_origin_list.append(originitem)
    
writejsonlist(name,write_path,test_origin_list)