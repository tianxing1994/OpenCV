import os


the_help = [188, 197, 194, 198, 185, 186, 194, 200, 179, 203]

dirname = 'dataset/nlp/data_digits/test'
train_dirname = 'dataset/nlp/data_digits/train'

file_list = os.listdir(dirname)

m = len(file_list)

for i in range(m):
    classify, suffix = file_list[i].split('_')
    classify = int(classify)
    the_help[classify] += 1
    number = the_help[classify]
    new_file_name = os.path.join(train_dirname, str(classify) + '_' + str(number) + '.txt')

    os.rename(os.path.join(dirname, file_list[i]), new_file_name)
