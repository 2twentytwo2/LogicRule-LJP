'''
    charge，数据集CJO22

'''

import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 必须放在torh 之前 选择机器
sys.path.append('/data2/zhZH/ljp-llm/') 
sys.path.append('/data2/zhZH/ljp-llm/utils')
sys.path.append('/data2/zhZH/ljp-llm/evaluate')
curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.path.append(curr_path)
sys.path.append(parent_path)
from utils.FileOp import File_op
from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluate.Evaluate_op import Evaluate_op
import re
import os
import random
import string
from sklearn.metrics import precision_recall_fscore_support

config = {
    "readroot" :'/data2/zhZH/ljp-llm/mydata/pljp/Bert_CJO22test_top10(new)_simcase.json',
    "smallmodel": "Bert",
    "crimelabelpath":"/data2/zhZH/ljp-llm/mydata/CAIL2018/crimeLabel.json",
    "crimerulepath":"",
    "writeroot": '/data2/zhZH/ljp-llm/log/pljp/llama3/Bert',
    'option_num' : 2,
    'test_len':0.4, # cail 0.005=749;cjo220.5=833;cjo0.4=666
    'TASK': { "article": 0, "crime":1, "penalty":2 }
}
acclogpath = f"{config['writeroot']}/charge_top5Upadaterulestp_CJOsimcase_acc.json"
dialogpath = config["writeroot"]+ '/charge_top5Upadaterulestp_CJOsimcase_dialog.json'
prompt_tempalte = '''作为中国法律领域的专家，请根据罪犯的案件事实判断案件是否符合罪名‘{crime}’的规则。
案件细节:
{fact}
罪名对应的法律如下:
{articletxts}
判断规则:
{rule}
是否符合，选项如下:
(A) yes
(B) no
可以结合以下两个相似案例进行思考，第一个案件中罪犯的罪名为{simcase1_article}；
第二个案件中罪犯的罪名为{simcase2_article}。
请一步步仔细思考。At the end show the answer option bracketed between <answer> and </answer>.
'''

abstract_prompt_tempalte = '''作为中国法律领域的专家，请根据下述案件细节中罪犯的行为进行摘要。\n
案件细节:
{fact}
如果案件细节中有存在的犯罪时间、地点、行为、犯罪结果、罪犯的主观心态等细节请汇总，最后生成少于3000字的摘要。At the end show the abstract bracketed between <abstract> and </abstract>.
'''

model_id = "zhichen/Llama3-Chinese"
model_dir = '/data2/zhZH/llamaChinese/Llama3-Chinese'

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype="auto", device_map="auto")


def writelog(mesage,result,label,dialogpath):
    jsonlog = dict()
    jsonlog["mesage"] = mesage
    jsonlog["response"] = result
    jsonlog["label"] = label
    Evaluate_op.writeJsonLog(jsonlog, dialogpath)


def getResponse(mesage):
    messages = [
            {"role": "system", "content": "You are a professional Chinese Law assistant. Please answer in Chinese."},
            {"role": "user", "content": mesage},
        ]
    input_ids = tokenizer.apply_chat_template(messages,add_generation_prompt=True,return_tensors="pt" ).to(model.device)

    terminators = [tokenizer.eos_token_id,tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    outputs = model.generate(
        input_ids,
        max_new_tokens=4096,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    result = tokenizer.decode(response, skip_special_tokens=True)
    return result

def getACC(index,truelist, predictlist):
    acc, _, _, _ = precision_recall_fscore_support(truelist, predictlist, average='micro')
    map, mar, maf, _ = precision_recall_fscore_support(truelist, predictlist, average='macro')

    acc = round(acc, 6) * 100
    map = round(map, 6) * 100
    mar = round(mar, 6) * 100
    maf = round(maf, 6) * 100

    log = { "index":index,"acc":acc, "map":map, "mar":mar, "maf":maf}
    return log

def analysisAbstractResponse(response):
    clean_pattern = r"<abstract>([\s\S]*?)<\/abstract>"
    match = re.findall(clean_pattern, response)
    if len(match) == 0:
        return response
    else:
        return match[-1]

def getAbstractFact(fact):
    mesage = abstract_prompt_tempalte.format( fact=fact)
    response = getResponse(mesage)
    new_fact = analysisAbstractResponse(response)
    return new_fact

# 分析回复
def analysisResponse(response,onetop):
    if '<answer>yes' in response:
        return onetop,False
    else:
        option_num = config['option_num']
        letters = string.ascii_uppercase[:option_num] + string.ascii_lowercase[:option_num]
        clean_pattern = r"<answer>([\s\S]*?)<\/answer>"
        match = re.findall(clean_pattern, response.lower())
        if len(match) == 0:
            return -1, True

        answer = re.search(r"\([" + letters + r"]\)", match[-1])
        if answer is not None:
            return_answer =  answer.group(0)[1].upper()
            if return_answer == 'A':
                return onetop, False
            else:
                return -1, True
        answer = re.search(r"[" + letters + r"]", match[-1])
        if answer is None:
            return -1, True
        
        return_answer = answer[0].upper()
        if return_answer == 'A':
            return onetop, False

        return -1, True

# 根据罪名名称返回对应法条的texts
# 输入 crime_name
# 根据字典 key:crime_name, item: article_json_list:[{num:xx,str:xx}]
# 返回 所有法条结合一起的str
def getArtcletxt(crimename):
    returnstr = ''
    if crimename in crime_articleDict.keys():
        article_jsonlist = crime_articleDict[crimename]
        single_a_prompt = '''
            第{aid}条法条：{astr}\n
        '''
        for ajson in article_jsonlist:
            aid = ajson['num']
            astr = ajson['str']
            signle_a = single_a_prompt.format(aid = aid, astr = astr)
            returnstr += signle_a
    
    return returnstr


# 5个候选项没有得出答案后
# 使用本方法在所有罪名中查询
def checkbyAllChargeRule(jsoncase):
    iscontinue = True
    for onetop in crimelablelist:
        fact=jsoncase['fact'].replace(' ','')
        article = jsoncase['article']
        if onetop not in crimelabel2Intdict.keys():
            continue
        cakey = (crimelabel2Intdict[onetop],article)
        if cakey not in ruledict.keys():
            continue
        rule,article_text = ruledict[cakey]
        crimename = onetop
        # fact过长
        if len(fact) > 5000:
            fact = getAbstractFact(fact)
        simcases = jsoncase['simcases']
        charge = jsoncase['charge']
        simcase1_article,simcase1_fact = simcases[0]['charge'],simcases[0]['fact']
        simcase2_article,simcase2_fact = simcases[1]['charge'],simcases[1]['fact']
        mesage = prompt_tempalte.format(fact=fact,articletxts=article_text,rule=rule,crime=crimename,simcase1_article=charge,simcase2_article=simcase2_article)

        response = getResponse(mesage)
        predictlabel, iscontinue = analysisResponse(response, onetop)
        
        writelog(mesage, response, predictlabel,dialogpath)
        if iscontinue != True: # 选择了 A. yes 符合
            break
    
    return predictlabel # 全no就是-1

def checkbyChargeRule(jsoncase, top5list):
    iscontinue = True
    for onetop in top5list:
        fact=jsoncase['fact'].replace(' ','')
        article = jsoncase['article']
        if onetop not in crimelabel2Intdict.keys():
            continue
        cakey = (crimelabel2Intdict[onetop],article)
        if cakey not in ruledict.keys():
            continue
        rule,article_text = ruledict[cakey]
        crimename = onetop
        # fact过长
        if len(fact) > 5000:
            fact = getAbstractFact(fact)
        simcases = jsoncase['simcases']
        charge = jsoncase['charge']
        simcase1_article,simcase1_fact = simcases[0]['charge'],simcases[0]['fact']
        simcase2_article,simcase2_fact = simcases[1]['charge'],simcases[1]['fact']
        mesage = prompt_tempalte.format(fact=fact, articletxts=article_text, rule=rule, crime=crimename,simcase1_article=charge,simcase2_article=simcase2_article)

        response = getResponse(mesage)
        predictlabel, iscontinue = analysisResponse(response, onetop)
        
        writelog(mesage, response, predictlabel,dialogpath)
        if iscontinue != True: 
            # 选择了 A. yes 符合,不继续了
            break
    
    if iscontinue == True: # 全选的 No
        predictlabel = checkbyAllChargeRule(jsoncase)

    return predictlabel


def all_elements_in_b(a_list, b_list):
    # 将B列表转换为集合
    b_set = set(b_list)
    # 都在B中返回true
    return set(a_list).issubset(b_set)

if __name__ == '__main__':
    global crimelablelist, crimelabel2Intdict, ruledict, crime_articleDict

    readpath = config["readroot"]
    alllawtext = File_op.readalllawtext()
    testjsoncase = File_op.readjsonlist(readpath)
    # crimelablelist, crimelabel2Intdict = File_op.readcrimelabellist()
    crimelablelist, crimelabel2Intdict = File_op.getlabel_cjo22(config['TASK']['crime'])
    ruledict = File_op.getrule_cjo22(config['TASK']['crime']) # key:crime, item:rulestr
    crime_articleDict = File_op.readCrimeArticleDict() # key:crime, item:article_jsonlist
    weijinci = ['毒品','冰毒','海洛因']

    episode_acc = []
    for train_e in range(4):
        index = -1
        truechargelabels,  predictchargelabels = [],[]
        # 抽取10%，测试5轮，最后取平均值
        num_elements_to_select = int(len(testjsoncase) * config['test_len'])
        print(f'testcase_len = {num_elements_to_select}')
        testcase = random.sample(testjsoncase, num_elements_to_select)
        for jsoncase in testcase:
            if jsoncase['charge'] not in jsoncase['top5_charge']:
                continue
            if any(substring in jsoncase['fact'].replace(' ','') for substring in weijinci):
                continue
            # if not all_elements_in_b(jsoncase['top5_charge'],crimelablelist):
            #     continue
            if len(jsoncase['fact'])>3000:
                continue
            index += 1
            episode_charge = []
            charges = jsoncase['top5_charge'][0:1]
            charges.extend([jsoncase['charge']])
            charges.extend(jsoncase['top5_charge'][2:])
            for i in range(10): # 使用10轮，llama3会出现回答不一致的情况，多次询问选择出现次数最多的答案
                # charge 
                answer = checkbyChargeRule(jsoncase,charges)
                episode_charge.append(answer)

            finalchargelabel = Evaluate_op.multiitem(episode_charge)
            if finalchargelabel not in crimelablelist:
                continue
            finalchargelabel = crimelabel2Intdict[finalchargelabel] # 变成int
            truechargelabel = crimelabel2Intdict[jsoncase["charge"]]
            truechargelabels.append(truechargelabel) # 这个是list
            predictchargelabels.append(finalchargelabel) # 文字对应成数字
            writelog(f"predict_charge {truechargelabel}",episode_charge, finalchargelabel,dialogpath)
            if index % 50 == 0:
                Evaluate_op.writeAnswer(f"charge {index} ", acclogpath, truechargelabels, predictchargelabels)
                
        Evaluate_op.writeAnswer(f"trainepisode={train_e}  charge {index}", acclogpath, truechargelabels, predictchargelabels)
        episode_acc.append(getACC(train_e, truechargelabels,predictchargelabels))
        writelog(f"predict_charge trainepisode={train_e}",episode_acc, [],dialogpath)

    acc, map, mar, maf = 0,0,0,0
    for logi in episode_acc:
        # log = { "index":index,"acc":acc, "map":map, "mar":mar, "maf":maf}
        acc += logi['acc']
        map += logi['map']
        mar += logi['mar']
        maf += logi['maf']
    acc = acc /5
    map = map /5
    mar = mar /5
    maf = maf /5
    finallog = { "finalindex":index,"finalacc":acc, "finalmap":map, "finalmar":mar, "finalmaf":maf}
    writelog(f"predict_charge 10_acc",episode_acc, finallog, acclogpath)

        



