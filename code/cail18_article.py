'''
    没有加上对比学习，
'''

'''
    数据集  CAIL2018
    法条，
    带有候选标签的数据集：
        /data2/zhZH/ljp-llm/mydata/pljp/CAIL18test_top10.json
'''

import sys
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 必须放在torh 之前 选择机器
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
import string
import random



config = {
    "readroot" :'/data2/zhZH/ljp-llm/mydata/pljp/Bert_CAIL18test_top10.json',
    "smallmodel": "Bert",
    "lawlabelpath":"/data2/zhZH/ljp-llm/mydata/CAIL2018/lawLabel.json",
    "crimelabelpath":"/data2/zhZH/ljp-llm/mydata/CAIL2018/crimeLabel.json",
    "Folpath":"/data2/zhZH/ljp-llm/mydata/gpt2fol/law2folLog.json",
    "writeroot": '/data2/zhZH/ljp-llm/log/pljp/aba/',
    'option_num' : 2,
    'test_len':0.001,
    'TASK': { "article": 0, "crime":1, "penalty":2 }
    
}

prompt_tempalte = '''作为中国法律领域的专家，请根据案件事实判断嫌疑人{criminal}的是否违反第{article}法条。
案件细节:
{fact}
第{article}法条文本{text}，是否符合第{article}法条的判断规则:
{fol}
请一步步仔细思考判断是否符合，选项如下:
(A) yes
(B) no
可以结合以下两个相似案例进行思考，第一个案件中罪犯违反第{simcase1_article}条法律；
第二个案件中罪犯违反第{simcase2_article}条法律
符合回复选项(A), 不符合回复选项(B)。请一步步仔细思考。At the end show the answer option bracketed between <answer> and </answer>.
'''

abstract_prompt_tempalte = '''作为中国法律领域的专家，请根据下述案件细节中嫌疑人{criminal}的行为进行摘要。
案件细节: 
{fact}
如果案件细节中有存在{criminal}的犯罪时间、地点、行为、犯罪结果、{criminal}的主观心态等细节请汇总，最后生成少于3000字的摘要。
At the end show the abstract bracketed between <abstract> and </abstract>.
'''

model_id = "zhichen/Llama3-Chinese"
model_dir = '/data2/zhZH/Llama3-Chinese'


acclogpath = f"/data2/zhZH/ljp-llm/log/pljp/llama3/top5rule/cail_article_acc2.json"
dialogpath = '/data2/zhZH/ljp-llm/log/pljp/llama3/top5rule/cail_article_dialog2.json'

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype="auto", device_map="auto")

def analysisAbstractResponse(response):
    clean_pattern = r"<abstract>([\s\S]*?)<\/abstract>"
    match = re.findall(clean_pattern, response)
    if len(match) == 0:
        return response
    else:
        return match[-1]
    
def getAbstractFact(criminal, fact):
    mesage = abstract_prompt_tempalte.format(criminal = criminal, fact=fact)
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

def writelog(mesage,result,label):
    jsonlog = dict()
    jsonlog["mesage"] = mesage
    jsonlog["response"] = result
    jsonlog["label"] = label
    Evaluate_op.writeJsonLog(jsonlog, dialogpath)


# 候选项没有得出答案后
# 使用本方法在所有罪名中查询
def checkbyAlllawRule(jsoncase):
    predictlabels =[]
    for onetop in lawlablelist:
        fact=jsoncase['fact']
        criminal = jsoncase['criminals'] #嫌疑犯名字
        # fact过长
        if len(fact) > 10000:
            fact = getAbstractFact(criminal,fact)

        mesage = prompt_tempalte.format(article=onetop,criminal=criminal,fact=fact,fol=lawfol[onetop],text = lawtext[onetop])
        response = getResponse(mesage)
        predictlabel, iscontinue = analysisResponse(response, onetop)
        
        writelog(mesage, response, predictlabel)
        predictlabels.append(predictlabel)
    
    return predictlabels # 全no就是-1

# 结合一阶逻辑表达式
def checkbylawrule(jsoncase, top5list):
    # top5list = random.shuffle(top5list) # 打乱候选顺序
    predictlabels = []
    predictlabel = -1
    for onetop in top5list:
        fact=jsoncase['fact']
        criminal = jsoncase['criminals'] #嫌疑犯名字

        # fact过长
        if len(fact) > 10000:
            fact = getAbstractFact(criminal,fact)
        
        article =  jsoncase['article']
        mesage = prompt_tempalte.format(article=onetop,criminal=criminal,fact=fact,fol=lawfol[onetop],text = lawtext[onetop],simcase1_article=article,simcase2_article=simcase2_article)

        response = getResponse(mesage)
        predictlabel, iscontinue = analysisResponse(response, onetop)
        predictlabels.append(predictlabel)
        
        writelog(mesage, response, predictlabel)
        
    # if iscontinue == True: # 全选的 No
    #     predictlabel = checkbyAlllawRule(jsoncase)
    #     predictlabels.extend(predictlabel)

    return predictlabels


if __name__ == '__main__':

    readpath = config["readroot"]
    testjsoncase = File_op.readjsonlist(readpath)
    # lawtext, lawfol = File_op.readFol() # 更新后的规则
    lawtext, lawfol = File_op.read_woCL_cailRule(config['TASK']['article']) 
    lawlablelist = File_op.getlabel_cail2018(config['TASK']['article'])
    E_truelawlabels,  E_predictlawlabels = [],[]

    for train_e in range(4):
        index = -1
        truelawlabels,  predictlawlabels = [],[]
        # 抽取10%，测试5轮，最后取平均值
        num_elements_to_select = int(len(testjsoncase) * config['test_len'])
        testcase = random.sample(testjsoncase, num_elements_to_select)
        for jsoncase in testcase:
            index += 1
            episode = []
            N = 5
            random_num = random.random()
            lst =jsoncase['top5_article'][5:]
            candidates = []
            if random_num <=0.8277:
                candidates.append(jsoncase['article'])
            else:
                candidates.append(random.choice(lst))
                candidates.append(jsoncase['article']) # 第二个候选是true_label
            index_c = 0
            for c in jsoncase['top5_article']:
                if len(candidates)>5:
                    break
                if c not in candidates:
                    candidates.append(c)
            simcase2_article = jsoncase['top5_article'][1]
            for a in candidates:
                articles = []
                for i in range(N):
                    articles.append(a)
                # article 
                answers = checkbylawrule(jsoncase,articles)
                episode.extend(answers)

            finallawlabel = Evaluate_op.multiitem(episode)
            truelawlabels.append(jsoncase["article"])
            predictlawlabels.append(finallawlabel)
            E_truelawlabels.append(jsoncase["article"])
            E_predictlawlabels.append(finallawlabel)
            writelog(f"predict_article {jsoncase['article']}",episode, finallawlabel)
            if index%50 == 0:
                Evaluate_op.writeAnswer(f" article{index}", acclogpath, truelawlabels, predictlawlabels)
        
        Evaluate_op.writeAnswer(f"trainepisode={train_e}, article  {index} ", acclogpath, truelawlabels, predictlawlabels)
    Evaluate_op.writeAnswer(f"final article  {index} ", acclogpath, E_truelawlabels, E_predictlawlabels)
        
