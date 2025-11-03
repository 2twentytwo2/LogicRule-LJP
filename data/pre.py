'''
    zy 根据data文件，分解出bert训练的数据
    原始数据
        训练集 /data2/zhZH/PLJP/data/precedent_database/precedent_case.json
        测试集 cail18和cjo22
            /data2/zhZH/PLJP/data/testset/cail18/testset.json
            /data2/zhZH/PLJP/data/testset/cjo22/testset.json

    数据结果
        .csv


    对比cjo22和cail18发现 法条标签中存在不同，cjo22多了以下12个法条标签，罪名标签包含。
        12，77 25 23 69 287 87 93 67 20 5 64 

'''
import random
import json
import csv

# 假设json文件名为"input.json"
json_file2 = "/data2/zhZH/ljp-llm/mydata/CAIL2018/data/data_test.json"
json_file = '/data2/zhZH/ljp-llm/data/CAIL2018_ALL_DATA/first_stage/test.json'
cjo22_file = "/data2/zhZH/PLJP/data/testset/cjo22/testset.json"
output_article = "/data2/zhZH/PLJP/data/csv/CAIL18/test_article.csv"
output_crime = "/data2/zhZH/PLJP/data/csv/CAIL18/test_crime.csv"
output_penalty = "/data2/zhZH/PLJP/data/csv/CAIL18/test_penalty.csv"
pt_cls2str = ["其他", "六个月以下", "六个月以上九个月以下", "九个月以上一年以下", "一年以上两年以下",
                    "两年以上三年以下", "三年以上五年以下", "五年以上七年以下",
                    "七年以上十年以下", "十年以上","死刑 or 无期"]
penaltydict = {}

def penalty2index(death,wuqi,month): # 可参考 ExtractCase.py penalty2str
    
    if death or wuqi:
        i = 10
    elif month > 10*12:
        i = 9
    elif month > 7*12:
        i = 8
    elif month > 5*12:
        i = 7
    elif month > 3*12:
        i = 6
    elif month > 2*12:
        i = 5
    elif month > 1*12:
        i = 4
    elif month > 9:
        i = 3
    elif month > 6:
        i = 2
    elif month > 0 :
        # print(f"刑期是{month}")
        i = 1
    else:
        i = 0

    return i


def labellist(path):
    articlelsit = []
    crimelsit = []
    with open(path, 'r') as file:
        for line in file:
            json_data = json.loads(line)
            relevant_articles = json_data['meta']['relevant_articles']
            accusation = json_data['meta']['accusation']
            if len(relevant_articles)>1 or len(accusation)>1 :
                continue
            relevant_articles = int(json_data['meta']['relevant_articles'][0])
            accusation = json_data['meta']['accusation'][0]

            if relevant_articles not in articlelsit:
                articlelsit.append(relevant_articles)
            if accusation not in crimelsit:
                crimelsit.append(accusation)
    print(f'len(articlelsit) ={len(articlelsit)}')
    print(f'len(crimelsit) ={len(crimelsit)}')
    return articlelsit,crimelsit

def comparelabels():
    cjo_a,cjo_c = labellist(cjo22_file)
    cail_a, cail_c = labellist(json_file)
    index_a = 0
    index_c = 0
    for a in cjo_a:
        if a not in cail_a:
            index_a +=1
            print(a)
    for c in cjo_c:
        if c not in cail_c:
            index_c+=1

    print(f'index_a = {index_a}')
    print(f'index_c = {index_c}')


def createcsv():
    # CAIL_a,CAIL_c = labellist(json_file)
    CJO22_a,CJO22_c = labellist(cjo22_file)

    # 读取json文件并写入csv文件
    with open(json_file, 'r') as file:
        # 创建csv写入器
        with open(output_article, 'w', newline='') as csv_article, open(output_crime, 'w', newline='') as csv_crime, open(output_penalty, 'w', newline='') as csv_penalty:
            writer1 = csv.writer(csv_article)
            writer2 = csv.writer(csv_crime)
            writer3 = csv.writer(csv_penalty)

            # 写入标题行
            writer1.writerow(['fact', 'label'])
            writer2.writerow(['fact', 'label'])
            writer3.writerow(['fact', 'label'])

            # 逐行读取json并写入csv
            for line in file:
                json_data = json.loads(line)
                fact = json_data['fact'].replace('\n','')
                fact = fact.replace('\r','')
                relevant_articles = json_data['meta']['relevant_articles']
                accusation = json_data['meta']['accusation']
                death = json_data['meta']['term_of_imprisonment']['death_penalty']
                wuqi = json_data['meta']['term_of_imprisonment']['life_imprisonment']
                month = json_data['meta']['term_of_imprisonment']['imprisonment']
                penalty = penalty2index(death,wuqi,month)
                # penalty = json_data['meta']['pt_cls'] # cjo22
                if len(relevant_articles)>1 or len(accusation)>1 :
                    continue
                relevant_articles = int(json_data['meta']['relevant_articles'][0])
                accusation = json_data['meta']['accusation'][0]

                if relevant_articles not in CJO22_a:
                    continue
                if accusation not in CJO22_c:
                    continue
                
                writer1.writerow([fact, relevant_articles])
                writer2.writerow([fact, accusation])
                writer3.writerow([fact, penalty])


def createcsv_cjo22train(): # 因为cjo的标签少，所以读取训练集时 需要考虑标签没有超过测试集
    # CAIL_a,CAIL_c = labellist(json_file)
    CJO22_a,CJO22_c = labellist(cjo22_file)
    
    read_file = '/data2/zhZH/PLJP/data/precedent_database/precedent_case.json'
    output_article = "/data2/zhZH/PLJP/data/csv/cjo_train/train_article.csv"
    output_crime = "/data2/zhZH/PLJP/data/csv/cjo_train/train_crime.csv"
    output_penalty = "/data2/zhZH/PLJP/data/csv/cjo_train/train_penalty.csv"

    # 读取json文件并写入csv文件
    with open(read_file, 'r') as file:
        # 创建csv写入器
        with open(output_article, 'w', newline='') as csv_article, open(output_crime, 'w', newline='') as csv_crime, open(output_penalty, 'w', newline='') as csv_penalty:
            writer1 = csv.writer(csv_article)
            writer2 = csv.writer(csv_crime)
            writer3 = csv.writer(csv_penalty)

            # 写入标题行
            writer1.writerow(['fact', 'label'])
            writer2.writerow(['fact', 'label'])
            writer3.writerow(['fact', 'label'])

            # 逐行读取json并写入csv
            for line in file:
                json_data = json.loads(line)
                fact = json_data['fact']
                relevant_articles = json_data['meta']['relevant_articles']
                accusation = json_data['meta']['accusation']
                death = json_data['meta']['term_of_imprisonment']['death_penalty']
                wuqi = json_data['meta']['term_of_imprisonment']['life_imprisonment']
                month = json_data['meta']['term_of_imprisonment']['imprisonment']
                penalty = penalty2index(death,wuqi,month)
                # penalty = json_data['meta']['pt_cls'] # cjo22
                if len(relevant_articles)>1 or len(accusation)>1 :
                    continue
                relevant_articles = int(json_data['meta']['relevant_articles'][0])
                accusation = json_data['meta']['accusation'][0]

                if relevant_articles not in CJO22_a:
                    continue
                if accusation not in CJO22_c:
                    continue
                
                writer1.writerow([fact, relevant_articles])
                writer2.writerow([fact, accusation])
                writer3.writerow([fact, penalty])


def createdata_CAIL():
    readpath = '/data2/zhZH/ljp-llm/data/CAIL2018_ALL_DATA/first_stage/test.json'
    writepath = '/data2/zhZH/ljp-llm/mydata/pljp/CAIL18_test.json'
    ljpdatas = []
    CJO22_a,CJO22_c = labellist(cjo22_file)
    with open(readpath, 'r') as file:
        with open(writepath, 'w', newline='') as writefile:
            for line in file:
                ljpdata = dict()
                json_data = json.loads(line)
                relevant_articles = json_data['meta']['relevant_articles']
                accusation = json_data['meta']['accusation']
                death = json_data['meta']['term_of_imprisonment']['death_penalty']
                wuqi = json_data['meta']['term_of_imprisonment']['life_imprisonment']
                month = json_data['meta']['term_of_imprisonment']['imprisonment']
                penalty = penalty2index(death,wuqi,month)
                penaltyStr = pt_cls2str[penalty]
                criminal = json_data['meta']['criminals'][0]
                if len(relevant_articles)>1 or len(accusation)>1 :
                    continue

                relevant_articles = int(json_data['meta']['relevant_articles'][0])
                accusation = json_data['meta']['accusation'][0]

                if relevant_articles not in CJO22_a:
                    continue
                if accusation not in CJO22_c:
                    continue

                fact = json_data['fact']
                penalty2strIndex = penalty
                ljpdata['fact'] = fact
                ljpdata['article'] = relevant_articles
                ljpdata['charge'] = accusation
                ljpdata['criminals'] = criminal
                ljpdata['penalty2strIndex'] = penalty2strIndex
                ljpdata['penaltyStr'] = penaltyStr
                ljpdata['penalty'] = month
                ljpdatas.append(ljpdata)
            
            ljpdata_str = json.dumps(ljpdatas, ensure_ascii=False, indent=4)
            writefile.write(ljpdata_str)
            print(f'len(ljpdatas) = {len(ljpdatas)}')


def createdata_CJO22():
    CAIL_a,CAIL_c = labellist(json_file)

    readpath = '/data2/zhZH/PLJP/data/testset/cjo22/testset.json'
    writepath = '/data2/zhZH/ljp-llm/mydata/pljp/CJO22_test.json'
    ljpdatas = []
    with open(readpath, 'r') as file:
        with open(writepath, 'a', newline='') as writefile:
            for line in file:
                ljpdata = dict()
                json_data = json.loads(line)
                relevant_articles = json_data['meta']['relevant_articles']
                accusation = json_data['meta']['accusation']
                penalty = json_data['meta']['penalty'] # cjo22
                pt_cls = json_data['meta']['pt_cls']
                penaltyStr = pt_cls2str[pt_cls]
                if len(relevant_articles)>1 or len(accusation)>1 :
                    continue

                relevant_articles = int(json_data['meta']['relevant_articles'][0])
                accusation = json_data['meta']['accusation'][0]

                if relevant_articles not in CAIL_a:
                    continue
                if accusation not in CAIL_c:
                    continue


                fact = json_data['fact']
                ljpdata['fact'] = fact
                ljpdata['article'] = relevant_articles
                ljpdata['charge'] = accusation
                ljpdata['penalty2strIndex'] = pt_cls
                ljpdata['penaltyStr'] = penaltyStr
                ljpdata['penalty'] = penalty
                ljpdatas.append(ljpdata)
            
            ljpdata_str = json.dumps(ljpdatas, ensure_ascii=False, indent=4)
            writefile.write(ljpdata_str)
            print(f'len(ljpdatas) = {len(ljpdatas)}')

def createjson_train():
    read_file = '/data2/zhZH/PLJP/data/precedent_database/precedent_case.json'
    out_trainjson = '/data2/zhZH/PLJP/data/csv/train/precedentdata_train.json'
    out_validjson = '/data2/zhZH/PLJP/data/csv/train/precedentdata_valid.json'
    alljsons = []
    train_jsons = []
    valid_jsons =[]

     # 读取json文件并写入csv文件
    with open(read_file, 'r') as file:
        # 逐行读取json并写入csv
        for line in file:
            json_data = json.loads(line)
            fact = json_data['fact']
            relevant_articles = json_data['meta']['relevant_articles']
            accusation = json_data['meta']['accusation']
            death = json_data['meta']['term_of_imprisonment']['death_penalty']
            wuqi = json_data['meta']['term_of_imprisonment']['life_imprisonment']
            month = json_data['meta']['term_of_imprisonment']['imprisonment']
            penalty = penalty2index(death,wuqi,month)
            # penalty = json_data['meta']['pt_cls'] # cjo22
            if len(relevant_articles)>1 or len(accusation)>1 :
                continue
            relevant_articles = int(json_data['meta']['relevant_articles'][0])
            accusation = json_data['meta']['accusation'][0]

            write_jsondata = {}
            write_jsondata['fact'] = fact
            write_jsondata['article'] = relevant_articles
            write_jsondata['charge'] = accusation
            write_jsondata['criminals'] =json_data['meta']['criminals']
            write_jsondata['penalty2strIndex'] =penalty
            write_jsondata['penalty'] = penalty

            alljsons.append(write_jsondata)

        train_select = int(len(alljsons) * 0.6)
        train_jsons = random.sample(alljsons, train_select)
        # valid_select = int(len(alljsons) * 0.4)
        valid_jsons = [item for item in alljsons if item not in train_jsons]

    try:
        with open(out_validjson, 'w', encoding='utf-8') as json_file:
            json.dump(valid_jsons, json_file, ensure_ascii=False, indent=4)
            json_file.write('\n')
        print(f"数据成功写入到 {out_validjson}")
    except Exception as e:
        print(f"写入文件时出现错误：{e}")

    try:
        with open(out_trainjson, 'w', encoding='utf-8') as json_file:
            json.dump(train_jsons, json_file, ensure_ascii=False, indent=4)
            json_file.write('\n')
        print(f"数据成功写入到 {out_validjson}")
    except Exception as e:
        print(f"写入文件时出现错误：{e}")


def createCAIL18json_train():
    read_file = '/data2/zhZH/ljp-llm/data/CAIL2018_ALL_DATA/restData/rest_data.json'
    out_trainjson = '/data2/zhZH/PLJP/data/csv/train/restdata_train2.json'
    out_validjson = '/data2/zhZH/PLJP/data/csv/train/restdata_valid2.json'
    alljsons = []
    train_jsons = []
    valid_jsons =[]
    _,CJO22_crims = labellist(cjo22_file)


     # 读取json文件并写入csv文件
    with open(read_file, 'r') as file:
        # 逐行读取json并写入csv
        for line in file:
            
            json_data = json.loads(line)
            fact = json_data['fact']
            relevant_articles = json_data['meta']['relevant_articles']
            accusation = json_data['meta']['accusation']
            death = json_data['meta']['term_of_imprisonment']['death_penalty']
            wuqi = json_data['meta']['term_of_imprisonment']['life_imprisonment']
            month = json_data['meta']['term_of_imprisonment']['imprisonment']
            penalty = penalty2index(death,wuqi,month)
            # penalty = json_data['meta']['pt_cls'] # cjo22
            if len(relevant_articles)>1 or len(accusation)>1 :
                continue
            relevant_articles = int(json_data['meta']['relevant_articles'][0])
            accusation = json_data['meta']['accusation'][0]
            
            if relevant_articles > 217:
                continue
            if relevant_articles <114:
                continue
            if accusation not in CJO22_crims:
                continue
            if len(alljsons) > 10000:
                break
            write_jsondata = {}
            write_jsondata['fact'] = fact
            write_jsondata['article'] = relevant_articles
            write_jsondata['charge'] = accusation
            write_jsondata['criminals'] =json_data['meta']['criminals']
            write_jsondata['penalty2strIndex'] =penalty
            write_jsondata['penalty'] = penalty

            alljsons.append(write_jsondata)

        train_select = int(len(alljsons) * 0.6)
        train_jsons = random.sample(alljsons, train_select)
        # valid_select = int(len(alljsons) * 0.4)
        rest_jsons = [item for item in alljsons if item not in train_jsons]
        valid_select = int(len(rest_jsons) * 0.4)
        valid_jsons = rest_jsons # random.sample(valid_select, valid_select)
        print(f'len_train = {len(train_jsons)}  len_vaild = {len(valid_jsons)}')

    try:
        with open(out_validjson, 'w', encoding='utf-8') as json_file:
            json.dump(valid_jsons, json_file, ensure_ascii=False, indent=4)
            json_file.write('\n')
        print(f"数据成功写入到 {out_validjson}")
    except Exception as e:
        print(f"写入文件时出现错误：{e}")

    try:
        with open(out_trainjson, 'w', encoding='utf-8') as json_file:
            json.dump(train_jsons, json_file, ensure_ascii=False, indent=4)
            json_file.write('\n')
        print(f"数据成功写入到 {out_validjson}")
    except Exception as e:
        print(f"写入文件时出现错误：{e}")

# 根据dladan 在cailtest生成数据 
def generateBert_CAIL_TEST():
    out_testpath = '/data2/zhZH/PLJP/data/csv/train/cail_test.json'
    read_file = '/data2/zhZH/ljp-llm/mydata/pljp/Bert_CAIL18test_top10.json'
    data_trainjson = '/data2/zhZH/PLJP/data/csv/train/data_train.json'
    data_validjson = '/data2/zhZH/PLJP/data/csv/train/data_valid.json'

    with open( data_validjson, 'r', encoding='utf-8') as file:
        validjson = []
        for line in file:
            data = json.loads(line)
            validjson.append(data)
    with open(data_trainjson, 'r', encoding='utf-8') as file:
        trainjson = []
        for line in file:
            data = json.loads(line)
            trainjson.append(data)
    with open(read_file, 'r', encoding='utf-8') as file:
        testjson = json.load(file)

    trainfacts = []
    articlelabels =[]
    crimelabels = []
    
    for json_data in trainjson:
        fact = json_data['fact']
        relevant_articles = json_data['article']
        accusation = json_data['charge']
        if len(relevant_articles)>1 or len(accusation)>1 :
            continue
        
        article = int(json_data['article'][0])
        charge = json_data['charge'][0]

        trainfacts.append(fact)
        if article not in articlelabels:
            articlelabels.append(article)
        if charge not in crimelabels:
            crimelabels.append(charge)
    
    for factjson in validjson:
        fact = factjson['fact']
        relevant_articles = factjson['article']
        accusation = factjson['charge']
        if len(relevant_articles)>1 or len(accusation)>1 :
            continue
        
        article = int(factjson['article'][0])
        charge = factjson['charge']

        trainfacts.append(fact)
        if article not in articlelabels:
            articlelabels.append(article)
        if charge not in crimelabels:
            crimelabels.append(charge)

    tests = []
    for a in testjson:
        factm = a['fact']
        charge = a['charge']
        article = a['article']
        if factm in trainfacts:
            continue
        if charge not in crimelabels and article not in articlelabels:
            continue
        if article > 217:
                continue
        if article <114:
            continue
        if article == 133 and len(tests) >= 100:
            continue
        #print(f'{article},,{charge},, {factm}')
        tests.append(a)
        if len(tests) > 10000:
            break
    print(f'len(tests) = {len(tests)}')
    try:
        with open(out_testpath, 'w', encoding='utf-8') as json_file:
            json.dump(tests, json_file, ensure_ascii=False, indent=4)
            json_file.write('\n')
        print(f"数据成功写入到 {out_testpath}")
    except Exception as e:
        print(f"写入文件时出现错误：{e}")



def getpenaltystr(penaltyint):
    pt_cls2str = ["其他", "六个月以下", "六到九个月", "九个月到一年", "一到两年", "二到三年", "三到五年", "五到七年", "七到十年", "十年以上","死刑","无期"]
    DEATH = -1
    WUQI = -2
    i = 0
    if penaltyint == 0:
        i = 0
    elif penaltyint > 10*12:
        i = 9
    elif penaltyint > 7*12:
        i = 8
    elif penaltyint > 5*12:
        i = 7
    elif penaltyint > 3*12:
        i = 6
    elif penaltyint > 2*12:
        i = 5
    elif penaltyint > 1*12:
        i = 4
    elif penaltyint > 9:
        i = 3
    elif penaltyint > 6:
        i = 2
    else:
        # print(f"刑期是{penaltyint}")
        i = 1

    return pt_cls2str[i],i

def generateBert_CAILexecrice_TEST():
    out_testpath = '/data2/zhZH/PLJP/data/csv/train/cail_test_exercise.json'
    read_file = '/data2/zhZH/ljp-llm/data/CAIL2018_ALL_DATA/exercise_contest/data_test.json'
    data_trainjson = '/data2/zhZH/PLJP/data/csv/train/data_train.json'
    data_validjson = '/data2/zhZH/PLJP/data/csv/train/data_valid.json'

    with open( data_validjson, 'r', encoding='utf-8') as file:
        validjson = []
        for line in file:
            data = json.loads(line)
            validjson.append(data)
    with open(data_trainjson, 'r', encoding='utf-8') as file:
        trainjson = []
        for line in file:
            data = json.loads(line)
            trainjson.append(data)
    with open(read_file, 'r', encoding='utf-8') as file:
        testjson = []
        for line in file:
            data = json.loads(line)
            testjson.append(data)

    trainfacts = []
    articlelabels =[]
    crimelabels = []
    
    for json_data in trainjson:
        fact = json_data['fact']
        relevant_articles = json_data['article']
        accusation = json_data['charge']
        if len(relevant_articles)>1 or len(accusation)>1 :
            continue
        
        article = int(json_data['article'][0])
        charge = json_data['charge'][0]

        trainfacts.append(fact)
        if article not in articlelabels:
            articlelabels.append(article)
        if charge not in crimelabels:
            crimelabels.append(charge)
    
    for factjson in validjson:
        fact = factjson['fact']
        relevant_articles = factjson['article']
        accusation = factjson['charge']
        if len(relevant_articles)>1 or len(accusation)>1 :
            continue
        
        article = int(factjson['article'][0])
        charge = factjson['charge']

        trainfacts.append(fact)
        if article not in articlelabels:
            articlelabels.append(article)
        if charge not in crimelabels:
            crimelabels.append(charge)

    tests = []
    for a in testjson:
        factm = a['fact']
        relevant_articles = a['meta']['relevant_articles']
        accusation = a['meta']['accusation']
        if len(relevant_articles)>1 or len(accusation)>1 :
            continue
        article = int(a['meta']['relevant_articles'][0])
        charge = a['meta']['accusation'][0]
        if factm in trainfacts:
            continue
        if charge not in crimelabels and article not in articlelabels:
            continue
        if article > 217:
                continue
        if article <114:
            continue
        if article == 133 and len(tests) >= 100:
            continue
        #print(f'{article},,{charge},, {factm}')

        change = {}
        change['fact'] = factm
        change['article'] = article
        change['charge'] = charge
        change['penalty'] = a['meta']['term_of_imprisonment']['imprisonment']
        strp,intp =getpenaltystr(change['penalty'])
        change['penaltystr'] = intp

        tests.append(change)
        if len(tests) > 10000:
            break
    print(f'len(tests) = {len(tests)}')
    try:
        with open(out_testpath, 'w', encoding='utf-8') as json_file:
            json.dump(tests, json_file, ensure_ascii=False, indent=4)
            json_file.write('\n')
        print(f"数据成功写入到 {out_testpath}")
    except Exception as e:
        print(f"写入文件时出现错误：{e}")

def checknocompare():
    test_path ='/data2/zhZH/ljp-llm/mydata/pljp/CJO22_test.json'
    data_trainjson = '/data2/zhZH/PLJP/data/csv/train/data_train.json'
    data_validjson = '/data2/zhZH/PLJP/data/csv/train/data_valid.json'

    with open( data_validjson, 'r', encoding='utf-8') as file:
        validjson = []
        for line in file:
            data = json.loads(line)
            validjson.append(data)
    with open(data_trainjson, 'r', encoding='utf-8') as file:
        trainjson = []
        for line in file:
            data = json.loads(line)
            trainjson.append(data)
    with open(test_path, 'r', encoding='utf-8') as file:
        testjson = json.load(file)

    trainfacts = []

    for factjson in trainjson:
        fact =factjson['fact']
        if fact not in trainfacts:
            trainfacts.append(fact)
    for factjson in validjson:
        fact =factjson['fact']
        if fact not in trainfacts:
            trainfacts.append(fact)

    for a in testjson:
        factm = a['fact']
        charge = a['charge']
        article = a['article']
        if factm in trainfacts:
            print(factm)


def find_top_5_percent(numbers):
    # 计算列表长度
    length = len(numbers)
    # 计算前 5%的元素数量
    top_5_count = int(length * 0.05)
    if top_5_count == 0:
        return []
    # 对列表进行排序
    sorted_numbers = sorted(numbers)
    # 提取前 5%大的数字
    top_5_percent = sorted_numbers[-top_5_count:]
    return top_5_percent

def get_CAILTEST_long():
    read_file = '/data2/zhZH/ljp-llm/mydata/pljp/Bert_CAIL18test_top10.json'
    data_trainjson = '/data2/zhZH/PLJP/data/csv/train/data_train.json'
    data_validjson = '/data2/zhZH/PLJP/data/csv/train/data_valid.json'

    with open( data_validjson, 'r', encoding='utf-8') as file:
        validjson = []
        for line in file:
            data = json.loads(line)
            validjson.append(data)
    with open(data_trainjson, 'r', encoding='utf-8') as file:
        trainjson = []
        for line in file:
            data = json.loads(line)
            trainjson.append(data)
    with open(read_file, 'r', encoding='utf-8') as file:
        testjson = json.load(file)

    trainfacts = []
    articlelabels =[]
    crimelabels = []
    
    for json_data in trainjson:
        fact = json_data['fact']
        relevant_articles = json_data['article']
        accusation = json_data['charge']
        if len(relevant_articles)>1 or len(accusation)>1 :
            continue
        
        article = int(json_data['article'][0])
        charge = json_data['charge'][0]

        trainfacts.append(fact)
        if article not in articlelabels:
            articlelabels.append(article)
        if charge not in crimelabels:
            crimelabels.append(charge)
    
    for factjson in validjson:
        fact = factjson['fact']
        relevant_articles = factjson['article']
        accusation = factjson['charge']
        if len(relevant_articles)>1 or len(accusation)>1 :
            continue
        
        article = int(factjson['article'][0])
        charge = factjson['charge']

        trainfacts.append(fact)
        if article not in articlelabels:
            articlelabels.append(article)
        if charge not in crimelabels:
            crimelabels.append(charge)

    tests_factlen =[]
    for a in testjson:
        factm = a['fact']
        len_factm = len(factm)
        tests_factlen.append(len_factm)

    result = find_top_5_percent(tests_factlen)
    len_normal = result[0]

    tests_long =[]
    for a in testjson:
        factm = a['fact']
        charge = a['charge']
        article = a['article']
        if factm in trainfacts:
            continue
        if charge not in crimelabels and article not in articlelabels:
            continue
        if article > 217:
                continue
        if article <114:
            continue
        if article == 133 and len(tests_long) >= 100:
            continue
        if len(factm) >=len_normal:
            tests_long.append(a)
        if len(tests_long) >= 5000:
            break
    print(f'len(tests) = {len(tests_long)}')

    out_path = '/data2/zhZH/PLJP/data/csv/train/cail_test_long.json'

    try:
        with open(out_path, 'w', encoding='utf-8') as json_file:
            json.dump(tests_long, json_file, ensure_ascii=False, indent=4)
            json_file.write('\n')
        print(f"数据成功写入到 {out_path}")
    except Exception as e:
        print(f"写入文件时出现错误：{e}")


def get_CAILTESTlong_baseCAILlong():
    read_file = '/data2/zhZH/ljp-llm/mydata/pljp/Bert_CAIL18test_top10_long.json'
    data_trainjson = '/data2/zhZH/PLJP/data/csv/train/data_train.json'
    data_validjson = '/data2/zhZH/PLJP/data/csv/train/data_valid.json'

    with open( data_validjson, 'r', encoding='utf-8') as file:
        validjson = []
        for line in file:
            data = json.loads(line)
            validjson.append(data)
    with open(data_trainjson, 'r', encoding='utf-8') as file:
        trainjson = []
        for line in file:
            data = json.loads(line)
            trainjson.append(data)
    with open(read_file, 'r', encoding='utf-8') as file:
        testjson = json.load(file)

    trainfacts = []
    articlelabels =[]
    crimelabels = []
    
    for json_data in trainjson:
        fact = json_data['fact']
        relevant_articles = json_data['article']
        accusation = json_data['charge']
        if len(relevant_articles)>1 or len(accusation)>1 :
            continue
        
        article = int(json_data['article'][0])
        charge = json_data['charge'][0]

        trainfacts.append(fact)
        if article not in articlelabels:
            articlelabels.append(article)
        if charge not in crimelabels:
            crimelabels.append(charge)
    
    for factjson in validjson:
        fact = factjson['fact']
        relevant_articles = factjson['article']
        accusation = factjson['charge']
        if len(relevant_articles)>1 or len(accusation)>1 :
            continue
        
        article = int(factjson['article'][0])
        charge = factjson['charge']

        trainfacts.append(fact)
        if article not in articlelabels:
            articlelabels.append(article)
        if charge not in crimelabels:
            crimelabels.append(charge)


    tests_long =[]
    for a in testjson:
        factm = a['fact']
        charge = a['charge']
        article = a['article']
        if factm in trainfacts:
            continue
        if charge not in crimelabels and article not in articlelabels:
            continue
        if article > 217:
                continue
        if article <114:
            continue
        if article == 133 and len(tests_long) >= 100:
            continue
        tests_long.append(a)

    print(f'len(tests) = {len(tests_long)}')

    out_path = '/data2/zhZH/PLJP/data/csv/train/cailLONG_test_long.json'

    try:
        with open(out_path, 'w', encoding='utf-8') as json_file:
            json.dump(tests_long, json_file, ensure_ascii=False, indent=4)
            json_file.write('\n')
        print(f"数据成功写入到 {out_path}")
    except Exception as e:
        print(f"写入文件时出现错误：{e}")

def getmax_min_article(labeldict):
    min_labels = [348,294,388,128]
    max_label,max_num = '',-1
    min_label,min_num = '', 999999
    for label,num in labeldict.items():
        if num>max_num:
            max_num =num
            max_label = label
        if num < min_num and label not in min_labels:
            min_num = num
            min_label = label
    
    print(f'max_num = {max_num}, maxlabel = {max_label}')
    print(f'min_num = {min_num}, min_label = {min_label}')

def getmax_min_charge(labeldict): # 最小68，目标100+; 非法狩猎 103
    min_labels = ['串通投标'] 
    max_label,max_num = '',-1
    min_label,min_num = '', 999999
    for label,num in labeldict.items():
        if num>max_num:
            max_num =num
            max_label = label
        if num < min_num and label not in min_labels:
            min_num = num
            min_label = label
    
    print(f'max_num = {max_num}, maxlabel = {max_label}')
    print(f'min_num = {min_num}, min_label = {min_label}')


def createdata_CAILforjunheng():
    readpath = '/data2/zhZH/ljp-llm/data/CAIL2018_ALL_DATA/first_stage/test.json'
    readpath2 = '/data2/zhZH/ljp-llm/data/CAIL2018_ALL_DATA/restData/rest_data.json'
    writepath = '/data2/zhZH/ljp-llm/mydata/pljp/CAIL18_test.json'
    ljpdatas = []
    labelsdict_a ={}
    labelsdict_c ={}
    json_datas = []
    CJO22_a,CJO22_c = labellist(cjo22_file)
    with open(readpath, 'r') as file:
        # with open(writepath, 'w', newline='') as writefile:
        for line in file:
            # ljpdata = dict()    
            json_data = json.loads(line)
            relevant_articles = json_data['meta']['relevant_articles']
            accusation = json_data['meta']['accusation']
            if len(relevant_articles)>1 or len(accusation)>1 :
                continue
            relevant_articles = int(json_data['meta']['relevant_articles'][0])
            accusation = json_data['meta']['accusation'][0]

            if relevant_articles not in CJO22_a:
                continue
            if accusation not in CJO22_c:
                continue
            json_datas.append(json_data)
    with open(readpath2, 'r') as file:
        # with open(writepath, 'w', newline='') as writefile:
        for line in file:
            # ljpdata = dict()    
            json_data = json.loads(line)
            relevant_articles = json_data['meta']['relevant_articles']
            accusation = json_data['meta']['accusation']
            if len(relevant_articles)>1 or len(accusation)>1 :
                continue
            relevant_articles = int(json_data['meta']['relevant_articles'][0])
            accusation = json_data['meta']['accusation'][0]

            if relevant_articles not in CJO22_a:
                continue
            if accusation not in CJO22_c:
                continue
            json_datas.append(json_data)

    for json_data in json_datas:
            relevant_articles = json_data['meta']['relevant_articles']
            accusation = json_data['meta']['accusation']
            death = json_data['meta']['term_of_imprisonment']['death_penalty']
            wuqi = json_data['meta']['term_of_imprisonment']['life_imprisonment']
            month = json_data['meta']['term_of_imprisonment']['imprisonment']
            penalty = penalty2index(death,wuqi,month)
            penaltyStr = pt_cls2str[penalty]
            criminal = json_data['meta']['criminals'][0]
            if len(relevant_articles)>1 or len(accusation)>1 :
                continue

            relevant_articles = int(json_data['meta']['relevant_articles'][0])
            accusation = json_data['meta']['accusation'][0]

            if relevant_articles not in CJO22_a:
                continue
            if accusation not in CJO22_c:
                continue

            if relevant_articles not in labelsdict_a.keys():
                labelsdict_a[relevant_articles] = 0
            labelsdict_a[relevant_articles] += 1
            if accusation not in labelsdict_c.keys():
                labelsdict_c[accusation] = 0
            labelsdict_c[accusation] +=1
            
            # fact = json_data['fact']
            # penalty2strIndex = penalty
            # ljpdata['fact'] = fact
            # ljpdata['charge'] = accusation
            # ljpdata['criminals'] = criminal
            # ljpdata['penalty2strIndex'] = penalty2strIndex
            # ljpdata['penaltyStr'] = penaltyStr
            # ljpdata['penalty'] = month
            # ljpdatas.append(ljpdata)
        
        # ljpdata_str = json.dumps(ljpdatas, ensure_ascii=False, indent=4)
        # writefile.write(ljpdata_str)
        # print(f'len(ljpdatas) = {len(ljpdatas)}')
    print('----labelsdict_a--------')
    getmax_min_article(labelsdict_a)
    # print('----labelsdict_c--------')
    # getmax_min_charge(labelsdict_c)




# generateBert_CAIL_TEST() # ok
# get_CAILTESTlong_baseCAILlong()
# createdata_CAILforjunheng()  # 均衡标签

generateBert_CAILexecrice_TEST()