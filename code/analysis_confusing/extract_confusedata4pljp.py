'''
    构建相似案件
    
    {
        "fact": "公诉机关指控，2014年9月16日15时20分许，被告人杨某某在沈阳市大东区新东四街市场内，因摆摊占位问题与被害人吴某某发生厮打，杨某某用拳将吴某某面部打伤，吴某某用脚将杨某某右腿踢伤。经辽宁仁和司法鉴定中心鉴定，吴某某鼻骨骨折合并右侧上颌骨额突骨折为轻伤二级，右侧眶内壁骨折为轻微伤；杨某某右大腿软组织挫伤为轻微伤。被告人杨某某于2014年10月20日被公安机关传唤到案。案发后，被告人杨某某已经与被害人吴某某达成和解协议，且取得吴某某的谅解。",
        "article": [
            234
        ],
        "charge": [
            "故意伤害"
        ],
        "criminals": [
            "杨某某"
        ],
        "penalty": 7,
        "penaltyStr": "六个月以上九个月以下",
        "penalty2strIndex": 2,
        "top5_article": [
            234,
            134,
            235,
            293,
            347
        ],
        "top5_charge": [
            "故意伤害",
            "故意杀人",
            "过失致人重伤",
            "过失致人死亡",
            "非法拘禁"
        ],
        "top5_penalty": [
            3,
            5,
            10,
            1,
            0
        ],
        "simicase1":{
            同上
            
        },
        "simicase2":{
            同上
            
        },
    },
    
'''
import os
import sys
import json
from copy import deepcopy

path='/data2/zhZH/ljp-llm/mydata/analysis_data/pnHurt-10-18-21-01_structured.jsonl'
name = 'hurtPLJP'

origin_path = '/data2/zhZH/ljp-llm/mydata/CAIL2018/data/BertCnn_test_top5.json'
write_path = '/data2/zhZH/ljp-llm/mydata/analysis_confuse4pljp'

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

for k in range(len(testdatalist)):
    originitem,writeitem,simcase1item,simcase2item = dict(), dict(), dict(),dict()
    
    testitem = testdatalist[k]
    targetfact = testitem['positive']['fact']
    originitem = find_origindata(targetfact,origindatalist)
    if originitem == None:
        continue
    simcase1fact = testitem['negatives'][0]['fact']
    simcase1item = find_origindata(simcase1fact,origindatalist)
    if simcase1item == None :
        simcase1item=dict()
        simcase1item['fact']=testitem['negatives'][0]['fact']
        simcase1item['charge']=[testitem['negatives'][0]['label']]
        
    if k+1 <len(testdatalist):
        simcase2fact = testdatalist[k+1]['positive']['fact']
        simcase2item = find_origindata(simcase2fact,origindatalist)
    else:
        simcase2fact = testdatalist[k-1]['positive']['fact']
        simcase2item = find_origindata(simcase2fact,origindatalist)
    
    writeitem = deepcopy(originitem)
    writeitem['simicase1']=simcase1item
    writeitem['simicase2']=simcase2item
    
    
    test_origin_list.append(writeitem)
    
writejsonlist(name,write_path,test_origin_list)