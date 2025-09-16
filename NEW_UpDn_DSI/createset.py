import json
import pickle as cPickle
from tqdm import tqdm
# dataroot = '/opt/data/private/sxd/data/vqa/'
# name = 'vqacp_v2_train_questions.json'
# file = dataroot + name
# print(file)
# s=0

# Q = [[] for _ in range(20)]
#
# with open(file) as f:
#     questions = json.load(f)
#
# for question in questions:
#
#     cls = int(question['class'])
#     Q[cls].append(question)
# print(len(questions))
# for i in range(1,15):
#     newset = ('/opt/data/private/sxd/data/vqa/CL/vqacp_v2_train_questions_%s.json' %i)
#     with open(newset, 'w') as json_file:
#         json.dump(Q[i], json_file, indent=4)
#     s+=len(Q[i])
# print(s)

dataroot = '/opt/data/private/sxd/data/vqa/cp-cache/'
name = 'train_target.pkl'
file = dataroot + name
cnt = 0
with open(file, 'rb') as f:
    answers = cPickle.load(f)

for k in range(1,15):
    newset = ('/opt/data/private/sxd/data/vqa/CL/vqacp_v2_train_questions_%s.json' %k)
    print(newset)

    with open(newset) as q:
        questions = json.load(q)
    print(questions[0])
    ans=[]

    questions.sort(key=lambda x: x['question_id'])
    answers.sort(key=lambda x: x['question_id'])
    print(len(questions), len(answers))

    j=0
    for question in tqdm(questions):
        for i in range(j,len(answers)):
            if answers[i]['question_id'] == question['question_id']:
                ans.append(answers[i])
                j=i
                break
    print(len(ans))
    cnt+=len(ans)
    print('total ans:', cnt)
    newpkl = ('/opt/data/private/sxd/data/vqa/CL/cp-cache/train_target_%s.pkl' %k)
    with open (newpkl, 'wb') as p:
        cPickle.dump(ans, p)

#[39503, 43183, 18908, 4308, 174315, 64406]
#438183