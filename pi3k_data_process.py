import csv

import random


def read_train_data():
    all_map = {}
    with open('data/pi3k.csv', 'r', newline='', encoding='utf-8') as file:
    
        csv_reader = csv.reader(file) 
        next(csv_reader, None)
        for row in csv_reader:
            all_map[row[0]] = [int(row[1]),int(row[2]),int(row[3]),int(row[4])]
    return all_map

def read_one(name,all_data):
    d = {}
    with open('data/pi3k_{}.csv'.format(name), 'r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader, None)
        for row in csv_reader:
            if(row[0] in all_data):
                continue
            d[row[0]] = int(row[1])
    return d

def split_map(input_map, n = [0.8,0.9]):
    keys = list(input_map.keys())

    random.shuffle(keys)

    total_keys = len(keys)
    first_split = int(total_keys * n[0])
    second_split = int(total_keys * n[1])

    first_part = {key: input_map[key] for key in keys[:first_split]}
    second_part = {key: input_map[key] for key in keys[first_split:second_split]}
    third_part = {key: input_map[key] for key in keys[second_split:]}

    return first_part, second_part, third_part


def merge_map(all,one,idx,no = True):
    if no:
        return one
    for a in all:
        if a not in one:
            one[a] = all[a][idx]
    return one
def write_csv(path_name,title,data):
    with open(path_name, 'w', newline='', encoding='utf-8') as file:

        csv_writer =  csv.writer(file)
        csv_writer.writerow(title)
        
        for smiles in data:
            row = [smiles]
            if(type(data[smiles]) is int):
                row.append(data[smiles])
            else:
                row.extend(data[smiles])
            csv_writer.writerow(row)
if __name__== '__main__':
    all_data = read_train_data()
    
    alpha = read_one("alpha",all_data)
    beta = read_one("beta",all_data)
    delta = read_one("delta",all_data)
    gamma = read_one("gamma",all_data)

    train_all,val_all,test_all = split_map(all_data,[0.9,1])

    write_csv("./processed_data/all_train.csv",['smiles',"alpha","beta","delta","gamma"],train_all)
    write_csv("./processed_data/all_test.csv",['smiles',"alpha","beta","delta","gamma"],val_all)
    write_csv("./processed_data/all_val.csv",['smiles',"alpha","beta","delta","gamma"],val_all)

    train_alpha,val_alpha,test_alpha = split_map(alpha)
    write_csv("./processed_data/alpha_train.csv",['smiles',"alpha"],merge_map(train_all,train_alpha,0,False))
    write_csv("./processed_data/alpha_val.csv",['smiles',"alpha"],merge_map(val_all,val_alpha,0))
    write_csv("./processed_data/alpha_test.csv",['smiles',"alpha"],merge_map(test_all,test_alpha,0))

    train_beta,val_beta,test_beta = split_map(alpha)

    write_csv("./processed_data/beta_train.csv",['smiles',"beta"],merge_map(train_all,train_beta,1,False))
    write_csv("./processed_data/beta_val.csv",['smiles',"beta"],merge_map(val_all,val_beta,1))
    write_csv("./processed_data/beta_test.csv",['smiles',"beta"],merge_map(test_all,test_beta,1))
    
    train_delta,val_delta,test_delta = split_map(delta)
    
    write_csv("./processed_data/delta_train.csv",['smiles',"delta"],merge_map(train_all,train_delta,2,False))
    write_csv("./processed_data/delta_val.csv",['smiles',"delta"],merge_map(val_all,val_delta,2))
    write_csv("./processed_data/delta_test.csv",['smiles',"delta"],merge_map(test_all,test_delta,2))

    train_gamma,val_gamma,test_gamma = split_map(gamma)

    write_csv("./processed_data/gamma_train.csv",['smiles',"gamma"],merge_map(train_all,train_gamma,3,False))
    write_csv("./processed_data/gamma_val.csv",['smiles',"gamma"],merge_map(val_all,val_gamma,3))
    write_csv("./processed_data/gamma_test.csv",['smiles',"gamma"],merge_map(test_all,test_gamma,3))
    
