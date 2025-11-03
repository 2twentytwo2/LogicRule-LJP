

import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 必须放在torh 之前 选择机器
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
import string
import random

config = {
    "readroot" :'/data2/zhZH/ljp-llm/mydata/pljp/Bert_CAIL18test_top10.json',
    "smallmodel": "Bert",
    "lawlabelpath":"/data2/zhZH/ljp-llm/mydata/CAIL2018/lawLabel.json",
    "crimelabelpath":"/data2/zhZH/ljp-llm/mydata/CAIL2018/crimeLabel.json",
    'penaltyStrLabels':["其他", "六个月以下", "六个月以上九个月以下", "九个月以上一年以下", "一年以上两年以下","两年以上三年以下", "三年以上五年以下", "五年以上七年以下","七年以上十年以下", "十年以上","死刑or无期"], # cjo22 没有死刑or无期的标签
    "penaltyrule":"/data2/zhZH/ljp-llm/mydata/CaseItem/init_penaltyrule_cail.json",
    "writeroot": '/data2/zhZH/ljp-llm/log/pljp/llama3/',
    'option_num' : 2,
    'test_len':0.5, # cjo22  0.5=833
    'TASK': { "article": 0, "crime":1, "penalty":2 },
}
acclogpath = f"{config['writeroot']}{config['smallmodel']}/penalty_top5rulestp_CAIL_acc.json"
dialogpath = config["writeroot"]+ f'{config["smallmodel"]}/penalty_top5rulestp_CAIL_dialog.json'

prompt_tempalte = '''作为中国法律领域的专家，请根据案件事实判断案件是否符合{penalty}刑期的一阶逻辑表达式。
案件细节:
{fact}
一阶逻辑表达式:
{rule}
表达式对应的法条文本如下:
{articletext}
是否符合，选项如下:
(A) yes
(B) no
请一步步仔细思考。At the end show the answer option bracketed between <answer> and </answer>.
'''

abstract_prompt_tempalte = '''作为中国法律领域的专家，请根据下述案件细节中罪犯的行为进行摘要。
案件细节: 
{fact}
如果案件细节中有存在犯罪时间、地点、行为、犯罪结果、和罪犯的主观心态等细节请汇总，最后生成少于3000字的摘要。At the end show the abstract bracketed between <abstract> and </abstract>.
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

def getAbstractFact(fact):
    mesage = abstract_prompt_tempalte.format(fact=fact)
    response = getResponse(mesage)
    new_fact = analysisAbstractResponse(response)
    return new_fact

def writelog(mesage,result,label):
    jsonlog = dict()
    jsonlog["mesage"] = mesage
    jsonlog["response"] = result
    jsonlog["label"] = label
    Evaluate_op.writeJsonLog(jsonlog, dialogpath)

# 根据article找对应的penalty
def getplabels(jarticle):
    plist = []
    for penalty,article in paRulelist.keys():
        if article == jarticle:
            plist.append(penalty)
    
    return plist

# 候选项没有得出答案后
# 使用本方法在所有当前法条对应的刑期选项中查询
def checkbyAlllawRule(fact,jsoncase):
    
    article = jsoncase['article']
    plabels = getplabels(article)
    print(f'第{article}条法律 找到的penalty个数是{len(plabels)}')
    predictlabels = []
    for onetop in plabels:
        pa_key = (onetop,article)
        rule,article_txt = paRulelist[pa_key]
        # fact过长
        if len(fact) > 5000:
            fact = getAbstractFact(fact)
        mesage = prompt_tempalte.format(fact=fact,rule=rule,articletext = article_txt,penalty = config['penaltyStrLabels'][onetop])
        response = getResponse(mesage)
        predictlabel, iscontinue = analysisResponse(response, onetop)
        predictlabels.append(predictlabel)
        
        writelog(mesage, response, predictlabel)
        # if iscontinue != True: # 选择了 A. yes 符合
        #     break
    
    return predictlabels # 全no就是-1

# 结合一阶逻辑表达式
def checkbylawrule(fact, jsoncase, top5list):
    # top5list = random.shuffle(top5list) # 打乱候选顺序
    is_pipei = False
    predictlabel = -1
    for onetop in top5list:
        article = jsoncase['article']
        pa_key = (onetop,article)
        if pa_key not in paRulelist.keys():
            continue
        is_pipei = True
        rule,article_txt = paRulelist[pa_key]

        mesage = prompt_tempalte.format(fact=fact,rule=rule,articletext = article_txt,penalty = config['penaltyStrLabels'][onetop])
        response = getResponse(mesage)
        predictlabel, iscontinue = analysisResponse(response, onetop)
        
        writelog(mesage, response, predictlabel)
        if iscontinue != True: # 选择了 A
            break
    # if iscontinue == True or is_pipei== False: # 全选的 No 或是没匹配上key
    # predictlabel2 = checkbyAlllawRule(jsoncase) # 选不选的 都check一下
    # if predictlabel2 != predictlabel:
    #     return predictlabel2

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
    lawtext = File_op.readalllawtext()
    lawlablelist = File_op.getlabel_cail2018(config['TASK']['penalty'])
    paRulelist = File_op.getrule_cail18(config['TASK']['penalty'])
    weijinci = ['毒品','冰毒','海洛因']
    episode_acc = []
    for train_e in range(4):
        index = -1
        truelawlabels,  predictlawlabels = [],[]
        # 抽取10%，测试5轮，最后取平均值
        num_elements_to_select = int(len(testjsoncase) * config['test_len'])
        testcase = random.sample(testjsoncase, num_elements_to_select)
        print(f'testcase_len = {num_elements_to_select}')
        for jsoncase in testcase:
            index += 1
            # if jsoncase['penalty'] not in jsoncase['top5_penalty']:
            #     continue
            if any(substring in jsoncase['fact'] for substring in weijinci):
                continue
            episode = []
            for i in range(3): # 多轮，llama3会出现回答不一致的情况，多次询问选择出现次数最多的答案
                fact = jsoncase['fact']
                # fact过长
                if len(fact) > 5000:
                    fact = getAbstractFact(fact)
                # answer = checkbylawrule(fact,jsoncase,jsoncase["top5_penalty"])
                answers = checkbyAlllawRule(fact,jsoncase)
                # episode.append(answer)
                episode.extend(answers)

            finallawlabel = Evaluate_op.multiitem(episode)
            truelawlabels.append(jsoncase["penalty2strIndex"])
            predictlawlabels.append(finallawlabel)
            writelog(f"predict_penalty {jsoncase['penalty2strIndex']}",episode, finallawlabel)
            if index%50 == 0:
                Evaluate_op.writeAnswer(f"penalty{index}", acclogpath, truelawlabels, predictlawlabels)
        
        Evaluate_op.writeAnswer(f"trainepisode={train_e}, penalty  {index} ", acclogpath, truelawlabels, predictlawlabels)
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
    writelog(f"predict_penalty final_acc",episode_acc, finallog)