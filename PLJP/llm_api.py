# -*- coding: utf-8 -*-

import openai
import os
import json
import threading
import time
from tqdm import tqdm
import argparse
from mycode.utils import loader
from mycode.utils import prompt_gen
import tiktoken
import requests
from openai import OpenAI

Yue_KEY = 'sk-vxL4L0mxtg7u2e88v1bQJqB7fzSKABNDkP3zMc66N2JQLsUd'
MY_KEY = Yue_KEY

OpenAIclient = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key = MY_KEY,
    base_url="https://api.chatanywhere.tech/v1"
    # base_url="https://api.chatanywhere.cn/v1"
)

# url = "http://10.15.82.10:8000/v1/chat/completions" 原pljp代码
url = 'https://api.chatanywhere.com.cn/v1/chat/completions'

# 设置代理
# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"

key_pool = ["turbo-3.5" ]
openai.api_key = key_pool[0]


def dav_response(text_list, model_name):
    enc = tiktoken.get_encoding("p50k_base")
    max_token = [len(enc.encode(t)) for t in text_list]
    max_token = 4096 - max(max_token)
    response = openai.Completion.create(
        model=model_name, prompt=text_list, max_tokens=max_token)  # greedy
    return response

def dav2_response(text_list):
    return dav_response(text_list, "text-davinci-002")

def dav3_response(text_list):
    return dav_response(text_list, "text-davinci-003")

def turbo_response(text_list):
    assert len(text_list) == 1, "gpt-3.5-turbo使用batch必须为1"
    text = text_list[0]
    enc = tiktoken.get_encoding("cl100k_base")
    max_token = 4096 - len(enc.encode(text)) - 500
    # (1608 in the messages, 2496 in the completion
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": text}], max_tokens=max_token, temperature=0)
    return response


def get_openai_response(data, in_context_contents, key_pool, args, meta):
    assert len(data) == len(in_context_contents) # 判断两个列表是否一致
    llm_response_function = {
        "text-davinci-002": dav2_response,
        "text-davinci-003": dav3_response,
        "gpt-3.5-turbo": turbo_response,
    }
    response_function = llm_response_function[args.model]  # 目标模型

    truncated_data = []
    '''
        主要功能：生成prompt
        输出obj[{"prompt","caseID"}]
    '''
    for i in tqdm(range(len(data)), desc="pre-process prompt"): # tqdm 进度条函数 
        case = data[i]
        fact = case["fact"].replace(" ", "") # raw fact 原始数据（事实描述） 去掉空格
        caseID = case["caseID"]
        if args.use_split_fact: # reframed fact
            fact = [case["fact_split"]["zhuguan"], case["fact_split"]["keguan"], case["fact_split"]["shiwai"]]
            fact = "。".join(fact) # 变成一串字符串‘zhuguanxxx。keguanxxx。shiwaixxxx。’

        max_len = 512
        fact = loader.truncate_text(fact, max_len=max_len)  # 截断长度
        prompt = prompt_gen.retrieved_label_option_fewshot(fact, in_context_contents[i], args,caseID)  # 获取prompt

        obj = {}
        obj["prompt"] = prompt
        obj["caseID"] = case["caseID"]
        truncated_data.append(obj)  
        # break

    '''
        开始训练
    '''
    print("Starting")
    i = 0
    batch = args.batch  # batch是什么含义？  每次训练的batch，默认是1
    while i < len(truncated_data):
        print(f"=== {i} / {len(data)} ===")
        prompt_list = truncated_data[i:i+batch]
        text_list = [t["prompt"] for t in prompt_list]
        print(text_list[0])
        try:
            response = response_function(text_list)
            resp_temp = response.copy()
            responses = []
            for run in range(len(prompt_list)):
                resp_temp["choices"] = [response["choices"][run]]
                resp_temp["caseID"] = prompt_list[run]["caseID"]
                responses.append(resp_temp.copy())
        except openai.error.RateLimitError as e:  # 异常报错
            e = repr(e)
            print(e)
            if "limit" in e:
                time.sleep(60)  # 限时了
            elif "quota" in e:
                if len(key_pool) == 0:
                    print("用光了key！")
                    exit()
                print(f"=== 当前key: {key_pool[0]} ===")
                openai.api_key = key_pool[0]
                key_pool = key_pool[1:]
                time.sleep(1)
            continue
        except openai.error.APIError as e:
            e = repr(e)
            print(e)
            time.sleep(60)
        except Exception as e:
            # https://platform.openai.com/docs/guides/error-codes/api-errors
            e = repr(e)
            print(e)
            exit()

        time.sleep(1)
        with open(args.output_path, "a+", encoding="utf-8") as f:
            for resp in responses:
                line = json.dumps(resp, ensure_ascii=False)
                f.write(line + "\n")
        i += batch
        # break


def run_llm(args):
    data = []

    with open(args.input_path, encoding="utf-8") as f:
        for idx, line in enumerate(f.readlines()): 
            case = json.loads(line)
            data.append(case)

    dumped_data = []
    # 读取“输出路径文件”里的数据？？
    with open(args.output_path, encoding="utf-8") as f:
        for line in f.readlines():
            dumped_data.append(line)

    # 去除“输出路径文件”里的数据（用长度来去除）
    data = data[len(dumped_data):] # 读取没有答案的测试用例
    print("to run data count: ", len(data))

    # load precedents  
    precedents = loader.load_precedent(args.precedent_pool_path, args.precedent_idx_path)
    precedents = precedents[len(dumped_data):]  # 先例 长度和data一致
    # #纯随机选择类案
    # precedents = [precedents[0]]*(len(data))


    # load [charge, article, penalty] topk prediction
    topk_label_option = loader.load_topk_option(args.topk_label_option_path)
    topk_label_option = topk_label_option[len(dumped_data):]

    # load relevant article definition
    if args.task == "article":
        article_definition = loader.load_retrieved_articles(args.topk_label_option_path, index_type="num")
        article_definition = article_definition[len(dumped_data):]
    else:
        article_definition = [["#"] for _ in range(len(data))] # 用 _ 指代临时变量
    
    in_context_contents = [{
        "precedents": precedents[i],
        "topk_label_option": topk_label_option[i],
        "article_definition": article_definition[i], 
    }
        for i in range(len(precedents))]  # 字典列表

    
    # # meta knowledge
    meta = {}
    # in_context_contents = [None]*len(data)
    get_openai_response(data, in_context_contents, key_pool, args, meta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # base setting
    parser.add_argument(
        '--model', default='text-davinci-003', help='text-davinci-002/3 & gpt-3.5-turbo')
    parser.add_argument(
        '--dataset', default="cjo22", help='[cail18, cjo22]')
    parser.add_argument(
        '--small_model', default="CNN", help='[CNN, TopJudge, ELE]')
    parser.add_argument(
        '--task', default="charge", help='[charge, article, penalty]')
    parser.add_argument(
        '--shot', default=3, help='few shots')
    parser.add_argument(
        '--batch', default=1, help='batch, reduce throughput cost')
    parser.add_argument(
        '--retriever', default="dense_retrieval", help="[bm25, dense_retrieval]")
    parser.add_argument(
        '--use_split_fact', default=False, help="[fact / fact_split]")
    
    # input: testset path, output: llm response path
    parser.add_argument(
        '--input_path', default="", help='testset path')
    parser.add_argument(
        '--output_path', default="", help='llm response path')
    
    # in-context contents path
    parser.add_argument(
        '--precedent_pool_path', default="", help='pool of precedent')
    parser.add_argument(
        '--precedent_idx_path', default="", help='index list of precedent')
    parser.add_argument(
        '--topk_label_option_path', default="", help='index list of topk label option')
    parser.add_argument(
        '--predicted_article_path', default="", help='index the predicted article to help predict charge and penalty')
    parser.add_argument(
        '--predicted_charge_path',default="",help='index the predicted charge to help predict penalty')
    
    
    args = parser.parse_args()
    # 输入路径
    if args.input_path == "":  
        prefix = f"data/testset/{args.dataset}/"
        if not args.use_split_fact:
            args.input_path = prefix + "testset.json"
        else:
            args.input_path = prefix + "testset_fact_split.json" 
    # 结果输出路径
    if args.output_path == "": 
        args.output_path = f"./data/output/llm_out/{args.dataset}/{args.small_model}/{args.task}/{args.shot}shot/{args.model}.json"
        if not os.path.exists(args.output_path): # 创建文件
            dirname = os.path.dirname(args.output_path)
            os.makedirs(dirname, exist_ok=True)
            with open(args.output_path, "w") as f:
                pass
    # 先例所在路径
    if args.precedent_pool_path == "":
        prefix = f"data/precedent_database/"
        if not args.use_split_fact:
            args.precedent_pool_path = prefix + "precedent_case.json"
        else:
            args.precedent_pool_path = prefix + "precedent_case_fact_split.json" 
    # 先例id
    if args.precedent_idx_path == "":
        prefix = f"data/output/domain_model_out/precedent_idx/{args.dataset}/{args.retriever}/"
        args.precedent_idx_path = prefix + f"precedent_idxs_{args.task}.json"

    # topk个标签
    if args.topk_label_option_path == "":
        args.topk_label_option_path = f"data/output/domain_model_out/candidate_label/{args.dataset}/{args.small_model}/{args.task}_topk.json"

    # 最终预测的法条
    if args.predicted_article_path == "":
        args.predicted_article_path = f"data/output/llm_out/{args.dataset}/{args.small_model}/article/{args.shot}shot/{args.model}.json"
    # 最终预测的判决
    if args.predicted_charge_path == "":
        args.predicted_charge_path = f"data/output/llm_out/{args.dataset}/{args.small_model}/charge/{args.shot}shot/{args.model}.json"
    run_llm(args)
