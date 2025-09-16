import json
import pickle as cPickle
dataroot = '/opt/data/private/sxd/data/vqa/'
name = 'vqacp_v2_train_questions.json'
file = dataroot + name

# dataroot = '/opt/data/private/sxd/data/vqa/cp-cache/'
# name = 'train_target.pkl'
# file = dataroot + name

print(file)
label=[0 for _ in range(20)]
with open(file) as f:
    questions = json.load(f)

for q in questions:
    if q['question'][:10] == 'What color':
        label[1] += 1
        q['class'] = '1'
    elif q['question'][:8] == 'How many':
        label[2] += 1
        q['class'] = '2'
    elif q['question'][:12] == 'What kind of' or q['question'][:13] == 'What kinds of' or q['question'][:12] == 'What type of':
        label[3] += 1
        q['class'] = '3'
    elif q['question'][:5] == 'Which':
        label[4] += 1
        q['class'] = '4'
    elif q['question'][:2] == 'Is' or q['question'][:3] == 'Are' or q['question'][:3] == 'Was' or q['question'][:4] == 'Were' or q['question'][:2] == 'IS':
        label[5] += 1
        q['class'] = '5'
    elif q['question'][:7] == 'What is' or q['question'][:8] == 'What are':
        label[6] += 1
        q['class'] = '6'
    elif q['question'][:6] == 'Why is':
        label[7] += 1
        q['class'] = '7'
    elif q['question'][:5] == 'Where':
        label[8] += 1
        q['class'] = '8'
    elif q['question'][:3] == 'How':
        label[9] += 1
        q['class'] = '9'
    elif q['question'][:3] == 'Why':
        label[10] += 1
        q['class'] = '10'
    elif q['question'][:3] == 'Who':
        label[11] += 1
        q['class'] = '11'
    elif q['question'][:4] == 'What':
        label[12] += 1
        q['class'] = '12'
    elif q['question'][:2] == 'Do' or q['question'][:4] == 'Does' or q['question'][:3] == 'Did':
        label[13] += 1
        q['class'] = '13'
    else:
        label[14] += 1
        q['class'] = '14'

print(label)
print(sum(label))


# with open('/opt/data/private/sxd/data/vqa/vqacp_v2_test_questions_new.json', 'w') as json_file:
#     json.dump(questions, json_file, indent=4)
