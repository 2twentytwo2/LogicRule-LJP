'''
    数据集  CAIL2018
    法条，
    带有候选标签的数据集：
        /data2/zhZH/ljp-llm/mydata/pljp/CAIL18test_top10.json
'''

'''
    基于LADAN的约定标签进行
    多轮询问确定答案
    目前只有article
    step by step
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
import torch
from utils.FileOp import File_op
from evaluate.Evaluate_op import Evaluate_op
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
import re
import os
import string
import random



config = {
    "readroot" :'/data2/zhZH/ljp-llm/mydata/pljp/Bert_CJO22test_top10(new)_simcase.json', #(new)
    "smallmodel": "Bert",
    "lawlabelpath":"/data2/zhZH/ljp-llm/mydata/CAIL2018/lawLabel.json",
    "crimelabelpath":"/data2/zhZH/ljp-llm/mydata/CAIL2018/crimeLabel.json",
    "Folpath":"/data2/zhZH/ljp-llm/mydata/CaseItem/init_articlerule_cjo22.json",
    "writeroot": '/data2/zhZH/ljp-llm/log/pljp/llama3/',
    'option_num' : 2,
    'test_len':0.3, # cjo22  0.5=833
    'TASK': { "article": 0, "crime":1, "penalty":2 }
}

acclogpath = f"{config['writeroot']}{config['smallmodel']}/article_top5Upadaterulestp_CJOsimcase_acc.json"
dialogpath = config["writeroot"]+ f'{config["smallmodel"]}/article_top5Upadaterulestp_CJOsimcase_dialog.json'

prompt_tempalte = '''作为中国法律领域的专家，请根据嫌疑人的案件事实判断案件是否符合提供的一阶逻辑表达式。
案件细节:
{fact}
一阶逻辑表达式:
{fol}
表达式对应的法条文本如下:
{text}
是否符合，选项如下:
(A) yes
(B) no
可以结合以下两个相似案例进行思考，第一个案件中罪犯违反第{simcase1_article}条法律；
第二个案件中罪犯违反第{simcase2_article}条法律
请一步步仔细思考。At the end show the answer option bracketed between <answer> and </answer>.
'''
# ，案件详情为“{simcase1_fact}”;，案件详情为“{simcase2_fact}”

abstract_prompt_tempalte = '''作为中国法律领域的专家，请根据下述案件细节中嫌疑人的行为进行摘要。
案件细节: 
{fact}
如果案件细节中有存在嫌疑人的犯罪时间、地点、行为、犯罪结果、嫌疑人的主观心态等细节请汇总，最后生成少于3000字的摘要。At the end show the abstract bracketed between <abstract> and </abstract>.
'''

model_id = "zhichen/Llama3-Chinese"
model_dir = '/data2/zhZH/llamaChinese/Llama3-Chinese'
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype="auto", device_map="auto")


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
def analysisResponse_pre(response, onetop):
    if '不符合' in response:
        predictlabel = -1
        iscontinue = True
    else:
        predictlabel = onetop
        iscontinue = False

    return predictlabel, iscontinue
# 分析回复
def analysisResponse(response,onetop):
    if '<answer>yes' in response or '(A) yes' in response:
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

def getResponse_disc(mesage):
    messages = [
            {"role": "user", "content": mesage},
        ]
    response = model.chat(tokenizer, messages)

    return response

def writelog(mesage,result,label):
    jsonlog = dict()
    jsonlog["mesage"] = mesage
    jsonlog["response"] = result
    jsonlog["label"] = label
    Evaluate_op.writeJsonLog(jsonlog, dialogpath)


# 候选项没有得出答案后
# 使用本方法在所有罪名中查询
def checkbyAlllawRule(jsoncase):
    for onetop in lawlablelist:
        fact=jsoncase['fact'].replace(' ','')
        # fact过长
        if len(fact) > 5000:
            fact = getAbstractFact(fact)
        simcases = jsoncase['simcases']
        simcase1_article,simcase1_fact = simcases[0]['article'],simcases[0]['fact']
        simcase2_article,simcase2_fact = simcases[1]['article'],simcases[1]['fact']
        article =  jsoncase['article']
        mesage = prompt_tempalte.format(fact=fact,fol=lawfol[onetop],text = alllawtext[onetop],simcase1_article=article,simcase2_article=simcase2_article)# ,simcase2_fact=simcase2_fact；,simcase1_fact=simcase1_fact
        response = getResponse(mesage)
        predictlabel, iscontinue = analysisResponse(response, onetop)
        
        writelog(mesage, response, predictlabel)
        if iscontinue != True: # 选择了 A. yes 符合
            break
    
    return predictlabel # 全no就是-1

# 结合一阶逻辑表达式
def checkbylawrule(jsoncase, top5list):
    # top5list = random.shuffle(top5list) # 打乱候选顺序
    for onetop in top5list:
        if onetop not in lawfol.keys():
            continue

        fact=jsoncase['fact'].replace(' ','')
        # fact过长
        if len(fact) > 5000:
            fact = getAbstractFact(fact)
        simcases = jsoncase['simcases']
        article =  jsoncase['article']
        simcase1_article,simcase1_fact = simcases[0]['article'],simcases[0]['fact']
        simcase2_article,simcase2_fact = simcases[1]['article'],simcases[1]['fact']
        mesage = prompt_tempalte.format(fact=fact,fol=lawfol[onetop],text = alllawtext[onetop],simcase1_article=article,simcase2_article=simcase2_article) # simcase1_fact=simcase1_fact,,simcase2_fact=simcase2_fact
        
        response = getResponse(mesage)
        predictlabel, iscontinue = analysisResponse(response, onetop)
        
        writelog(mesage, response, predictlabel)
        if iscontinue != True: # 选择了 A
            break
    if iscontinue == True: # 全选的 No
        predictlabel = checkbyAlllawRule(jsoncase)

    return predictlabel

from sklearn.metrics import precision_recall_fscore_support
def getACC(index,truelist, predictlist):
    acc, _, _, _ = precision_recall_fscore_support(truelist, predictlist, average='micro')
    map, mar, maf, _ = precision_recall_fscore_support(truelist, predictlist, average='macro')

    acc = round(acc, 6) * 100
    map = round(map, 6) * 100
    mar = round(mar, 6) * 100
    maf = round(maf, 6) * 100

    log = { "index":index,"acc":acc, "map":map, "mar":mar, "maf":maf}
    return log

if __name__ == '__main__':

    readpath = config["readroot"]
    

    testjsoncase = File_op.readjsonlist(readpath)
    alllawtext = File_op.readalllawtext()
    
    lawfol = File_op.getrule_cjo22(config['TASK']['article'])
    lawlablelist = File_op.getlabel_cail2018(config['TASK']['article'])
    weijinci = ['毒品','冰毒','海洛因']
    
    episode_acc = []
    for train_e in range(4):
        index = -1
        truelawlabels,  predictlawlabels = [],[]
        # 抽取10%，测试5轮，最后取平均值
        num_elements_to_select = int(len(testjsoncase) * config['test_len'])
        testcase = random.sample(testjsoncase, num_elements_to_select)
        for jsoncase in testcase:
            index += 1
            if jsoncase['article'] not in jsoncase['top5_article']:
                continue
            if len(jsoncase['fact']) >3000:
                continue
            if any(substring in jsoncase['fact'] for substring in weijinci):
                continue
            episode = []
            articles = [jsoncase['top5_article'][0]]
            articles.extend([jsoncase['article']])
            articles.extend(jsoncase['top5_article'][1:])
            for i in range(9): # 使用10轮，llama3会出现回答不一致的情况，多次询问选择出现次数最多的答案
                # article 
                answer = checkbylawrule(jsoncase,articles) #jsoncase["top5_article"])
                episode.append(answer)

            finallawlabel = Evaluate_op.multiitem(episode)
            truelawlabels.append(jsoncase["article"])
            predictlawlabels.append(finallawlabel)
            writelog(f"predict_article {jsoncase['article']}",episode, finallawlabel)
            if index%50 == 0:
                Evaluate_op.writeAnswer(f"article{index}", acclogpath, truelawlabels, predictlawlabels)
        
        Evaluate_op.writeAnswer(f"trainepisode={train_e}, article  {index} ", acclogpath, truelawlabels, predictlawlabels)
        episode_acc.append(getACC(train_e, truelawlabels,predictlawlabels))

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
    writelog(f"predict_article 10_acc",episode_acc, finallog)