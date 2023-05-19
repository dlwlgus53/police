import json
import pdb
import random
from collections import defaultdict
random.seed(1)
data = json.load(open("./all_data.json", "r"))
random.shuffle(data)
split_idx = int(len(data)/10)
train_data = data[split_idx:]
dev_data = data[:split_idx]

with open('train_data.json', 'w') as f: json.dump(train_data, f, ensure_ascii=False, indent=4)
with open('dev_data.json', 'w') as f: json.dump(dev_data, f, ensure_ascii=False, indent=4)

# 683 dataset
# average turn : 3.0
# domains : defaultdict(<class 'int'>, {'검경금': 610, '기타': 16, '대출상담': 39, nan: 17, '가족지인사칭': 1})

turns = 0

domains = defaultdict(int)
turn_num = defaultdict(int)

for dial in data:
    turn_num[len(dial['log'])] +=1
    domains[dial['domain']] +=1   
    
print(f"turn_num : {turn_num}")
print(f"domains : {domains}")

for item in dict(turn_num).items():
    print(f"{item[0]} : {item[1]}")