with open('spml/datasets/voc12/seambox_train+_a6_th0.5_hed.txt', 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    new_line = line.replace('seambox_a6_th0.5', 'liger_orig')
    new_lines.append(new_line)

with open('spml/datasets/voc12/train+liger_hed.txt', 'w') as f:
    f.writelines(new_lines)