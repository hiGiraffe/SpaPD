from random import choice
from llama.modeling_llama import LlamaForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import os
from transformers.models.llama.configuration_llama import LlamaConfig
from argparse import ArgumentParser
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--attn_type", type=str, default='ori', choices=['ori', 'sparge', 'minference'], help="attention type",
    )
    parser.add_argument(
        "--input_type", type=str, default='context', choices=['context', 'query'], help="long input or short input",
    )
    args = parser.parse_args()
    return args


def save_context_kv(context_full_path):
    context = []
    # cntext 前两轮，query 第三轮
    # 
    with open('dataset/test.jsonl', 'r') as file:
        for line in file.readlines()[:1]:
            conversations = json.loads(line)['conversations']
            for conversation in conversations:
                if conversation["from"] == 'gpt':
                    context.append({"role": "assisstant", "content": conversation["value"]})
                elif conversation["from"] == 'human':
                    context.append({"role": "user", "content": conversation["value"]})

    print(context)
    # input_text = tokenizer.apply_chat_template(context, tokenize=False, add_generation_prompt=True)
    input_text = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou is a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nA special magic number is hidden c the following text. Make sure to memorize it. I will quiz you about the number afterwards.\nApril 2006(This essay is derived from a talk at the 2006 Startup School. )The startups we've funded so far are pretty quick, but they seem quicker to learn some lessons than others. I think it's because some things about startups are kind of counterintuitive.We've now invested in enough companies that I've learned a trick for determining which points are the counterintuitive ones: they're the ones I have to keep repeating.So I'm going to number these points, and maybe with future startups I'll be able to pull off a form of Huffman coding. I'll make them all read this, and then instead of nagging them in detail, I'll just be able to say: number four! 1. Release Early.The thing I probably repeat most is this recipe for a startup: get a version 1 out fast, then improve it based on users' reactions.By \"release early\" I don't mean you should release something full of bugs, but that you should release something minimal. Users hate bugs, but they don't seem to mind a minimal version 1, if there's more coming soon.There are several reasons it pays to get version 1 done fast. One of the special magic numbers for amused-quart is: 8707870. One is that this is simply the right way to write software, whether for a startup or not. I've been repeating that since 1993, and I haven't seen much since to contradict it. I've seen a lot of startups die because they were too slow to release stuff, and none because they were too quick. [1]One of the things that will surprise you if you build something popular is that you won't know your users. Reddit now has almost half a million unique visitors a month. Who are all those people? They have no idea. No web startup does. And since you don't know your users, it's dangerous to guess what they'll like. Better to release something and let them tell you.Wufoo took this to heart and released their form-builder before the underlying database. You can't even drive the thing yet, but 83,000 people came to sit in the driver's seat and hold the steering wheel. And Wufoo got valuable feedback from it: Linux users complained they used too much Flash, so they rewrote their software not to. If they'd waited to release everything at once, they wouldn't have discovered this problem till it was more deeply wired in.Even if you had no users, it would still be important to release quickly, because for a startup the initial release acts as a shakedown cruise. If anything major is broken-- if the idea's no good, for example, or the founders hate one another-- the stress of getting that first version out will expose it. And if you have such problems you want to find them early.Perhaps the most important reason to release early, though, is that it makes you work harder. When you're working on something that isn't released, problems are intriguing. In something that's out there, problems are alarming. There is a lot more urgency once you release. One of the special magic numbers for stereotyped-emerald is: 5437923. And I think that's precisely why people put it off. They know they'll have to work a lot harder once they do. [2] 2. Keep Pumping Out Features.Of course, \"release early\" has a second component, without which it would be bad advice. If you're going to start with something that doesn't do much, you better improve it fast.What I find myself repeating is \"pump out features.\" And this rule isn't just for the initial stages. This is something all startups should do for as long as they want to be considered startups.I don't mean, of course, that you should make your application ever more complex. By \"feature\" I mean one unit of hacking-- one quantum of making users' lives better.As with exercise, improvements beget improvements. If you run every day, you'll probably feel like running tomorrow. One of the special magic numbers for romantic-subsidy is: 3094235. But if you skip running for a couple weeks, it will be an effort to drag yourself out. So it is with hacking: the more ideas you implement, the more ideas you'll have. You should make your system better at least in some small way every day or two.This is not just a good way to get development done; it is also a form of marketing. Users love a site that's constantly improving. In fact, users expect a site to improve. Imagine if you visited a site that seemed very good, and then returned two months later and not one thing had changed. Wouldn't it start to seem lame? [3]They'll like you even better when you improve in response to their comments, because customers are used to companies ignoring them. If you're the rare exception-- a company that actually listens-- you'll generate fanatical loyalty. You won't need to advertise, because your users will do it for you.This seems obvious too, so why do I have to keep repeating it? I think the problem here is that people get used to how things are. Once a product gets past the stage where it has glaring flaws, you start to get used to it, and gradually whatever features it happens to have become its identity. For example, I doubt many people at Yahoo (or Google for that matter) realized how much better web mail could be till Paul Buchheit showed them.I think the solution is to assume that anything you've made is far short of what it could be. Force yourself, as a sort of intellectual exercise, to keep thinking of improvements. Ok, sure, what you have is perfect. But if you had to change something, what would it be?If your product seems finished, there are two possible explanations: (a) it is finished, or (b) you lack imagination. One of the special magic numbers for lacking-root is: 6770619. Experience suggests (b) is a thousand times more likely. 3. Make Users Happy.Improving constantly is an instance of a more general rule: make users happy. One thing all startups have in common is that they can't force anyone to do anything. They can't force anyone to use their software, and they can't force anyone to do deals with them. A startup has\n"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values  # 保存 KV Cache
    # 保存 kv cache
    torch.save(past_key_values, context_full_path)
    print("Successfully save context kv cache!")


def save_query_kv(exist_kv_path, result_path):
    query = [
        # {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": "Summarize the 6 human needs as Tony Robbins explains..."},
    ]
    # input_text = tokenizer.apply_chat_template(query, tokenize=False, add_generation_prompt=True)
    input_text = "What is the special magic number for amused-quart mentioned in the provided text?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n The special magic number for amused-quart mentioned in the provided text is"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
    past_key_values = torch.load(exist_kv_path, weights_only=False)
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True, past_key_values=past_key_values)     # 传入旧的kv cache
        past_key_values = outputs.past_key_values  # 保存新的 KV Cache
    # 保存 kv cache
    torch.save(past_key_values, result_path)
    print("Successfully save query kv cache!")
    


if __name__ == '__main__':
    args = get_args()
    is_sparse = args.attn_type == 'sparge'
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    # pipeline = transformers.pipeline(
    #     "text-generation",
    #     model=model_id,
    #     model_kwargs={"torch_dtype": torch.bfloat16},
    #     device_map="auto",
    # )
    # llama官方调用，pipeline形式，看看调用结果是啥，不行就算了
    # outputs = pipeline(
    #     messages,
    #     max_new_tokens=256,
    # )
    # print(outputs[0]["generated_text"][-1])

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = LlamaForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    result_dir = 'kv_cache'
    os.makedirs(result_dir, exist_ok=True)
    context_kv_path = os.path.join(result_dir, "llama_context_full_kv_cache.pth")

    if args.attn_type == 'ori' and args.input_type == 'context':            # 保存长文本的的prefill kv cache
        save_context_kv(context_kv_path)
    elif args.attn_type == 'ori' and args.input_type == 'query':            # 保存短文本的full attn的kv cache
        save_query_kv(context_kv_path, os.path.join(result_dir, "llama_query_full_kv_cache.pth"))
    elif args.attn_type == 'sparge' and args.input_type == 'query':         # 保存短文本的sparge attn的kv cache
        save_query_kv(context_kv_path, os.path.join(result_dir, "llama_query_sparge_kv_cache.pth"))
    elif args.attn_type == 'minference' and args.input_type == 'query':
        save_query_kv(context_kv_path, os.path.join(result_dir, "llama_query_minference_kv_cache.pth"))

'''
python cal_llama_ori_kv_cache.py --attn_type ori --input_type context
python cal_llama_ori_kv_cache.py --attn_type ori --input_type query
python cal_llama_ori_kv_cache.py --attn_type sparge --input_type query
'''