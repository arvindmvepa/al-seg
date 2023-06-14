num_clusters = 10
emb="_mae"
tag="_top3_anns"

if num_clusters:
    replace_string = f'liger_num_clusters{num_clusters}{emb}{tag}' 
    new_file = f'spml/datasets/voc12/train+liger_num_clusters{num_clusters}{emb}{tag}_hed.txt'
else:
    replace_string = 'liger_orig'
    new_file = 'spml/datasets/voc12/train+liger_hed.txt'

with open('spml/datasets/voc12/seambox_train+_a6_th0.5_hed.txt', 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    new_line = line.replace('seambox_a6_th0.5', replace_string)
    new_lines.append(new_line)

with open(new_file, 'w') as f:
    f.writelines(new_lines)