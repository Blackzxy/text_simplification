
### train dataset
import os
cnt=0
lst = []
for line in open("C:/Users/z1325/Desktop/NLP_Torch/text-summarization/GitHub_T5/resources/datasets/wiki/wiki.train.complex", encoding='utf-8'):   # 逐行读取
    print(line)
    lst.append(line)
    cnt += 1
    if cnt>5: 
        break

if os.path.exists("C:/Users/z1325/Desktop/NLP_Torch/text-summarization/GitHub_T5/resources/datasets/wiki/wiki.train_small.complex"):
    os.remove("C:/Users/z1325/Desktop/NLP_Torch/text-summarization/GitHub_T5/resources/datasets/wiki/wiki.train_small.complex")

file_write_obj = open("C:/Users/z1325/Desktop/NLP_Torch/text-summarization/GitHub_T5/resources/datasets/wiki/wiki.train_small.complex", 'w', encoding='utf-8')   # 新文件
for var in lst:
    file_write_obj.write(var)  
file_write_obj.close()


cnt=0
lst = []
for line in open("C:/Users/z1325/Desktop/NLP_Torch/text-summarization/GitHub_T5/resources/datasets/wiki/wiki.train.simple", encoding='utf-8'):   # 逐行读取
    print(line)
    lst.append(line)
    cnt += 1
    if cnt>5: 
        break


if os.path.exists("C:/Users/z1325/Desktop/NLP_Torch/text-summarization/GitHub_T5/resources/datasets/wiki/wiki.train_small.simple"):
    os.remove("C:/Users/z1325/Desktop/NLP_Torch/text-summarization/GitHub_T5/resources/datasets/wiki/wiki.train_small.simple")

file_write_obj = open("C:/Users/z1325/Desktop/NLP_Torch/text-summarization/GitHub_T5/resources/datasets/wiki/wiki.train_small.simple", 'w', encoding='utf-8')   # 新文件
for var in lst:
    file_write_obj.write(var) 

file_write_obj.close()
################### tuning dataset
import os
cnt=0
lst = []
for line in open("C:/Users/z1325/Desktop/NLP_Torch/text-summarization/GitHub_T5/resources/datasets/wiki/wiki.tuning.complex", encoding='utf-8'):   # 逐行读取
    print(line)
    lst.append(line)
    cnt += 1
    if cnt>5: 
        break

if os.path.exists("C:/Users/z1325/Desktop/NLP_Torch/text-summarization/GitHub_T5/resources/datasets/wiki/wiki.tuning_small.complex"):
    os.remove("C:/Users/z1325/Desktop/NLP_Torch/text-summarization/GitHub_T5/resources/datasets/wiki/wiki.tuning_small.complex")

file_write_obj = open("C:/Users/z1325/Desktop/NLP_Torch/text-summarization/GitHub_T5/resources/datasets/wiki/wiki.tuning_small.complex", 'w', encoding='utf-8')   # 新文件
for var in lst:
    file_write_obj.write(var)  
file_write_obj.close()


cnt=0
lst = []
for line in open("C:/Users/z1325/Desktop/NLP_Torch/text-summarization/GitHub_T5/resources/datasets/wiki/wiki.tuning.simple", encoding='utf-8'):   # 逐行读取
    print(line)
    lst.append(line)
    cnt += 1
    if cnt>5: 
        break


if os.path.exists("C:/Users/z1325/Desktop/NLP_Torch/text-summarization/GitHub_T5/resources/datasets/wiki/wiki.tuning_small.simple"):
    os.remove("C:/Users/z1325/Desktop/NLP_Torch/text-summarization/GitHub_T5/resources/datasets/wiki/wiki.tuning_small.simple")

file_write_obj = open("C:/Users/z1325/Desktop/NLP_Torch/text-summarization/GitHub_T5/resources/datasets/wiki/wiki.tuning_small.simple", 'w', encoding='utf-8')   # 新文件
for var in lst:
    file_write_obj.write(var) 

file_write_obj.close()
######################## test dataset
import os
cnt=0
lst = []
for line in open("C:/Users/z1325/Desktop/NLP_Torch/text-summarization/GitHub_T5/resources/datasets/wiki/wiki.test.complex", encoding='utf-8'):   # 逐行读取
    print(line)
    lst.append(line)
    cnt += 1
    if cnt>5: 
        break

if os.path.exists("C:/Users/z1325/Desktop/NLP_Torch/text-summarization/GitHub_T5/resources/datasets/wiki/wiki.test_small.complex"):
    os.remove("C:/Users/z1325/Desktop/NLP_Torch/text-summarization/GitHub_T5/resources/datasets/wiki/wiki.test_small.complex")

file_write_obj = open("C:/Users/z1325/Desktop/NLP_Torch/text-summarization/GitHub_T5/resources/datasets/wiki/wiki.test_small.complex", 'w', encoding='utf-8')   # 新文件
for var in lst:
    file_write_obj.write(var)  
file_write_obj.close()


cnt=0
lst = []
for line in open("C:/Users/z1325/Desktop/NLP_Torch/text-summarization/GitHub_T5/resources/datasets/wiki/wiki.test.simple", encoding='utf-8'):   # 逐行读取
    print(line)
    lst.append(line)
    cnt += 1
    if cnt>5: 
        break


if os.path.exists("C:/Users/z1325/Desktop/NLP_Torch/text-summarization/GitHub_T5/resources/datasets/wiki/wiki.test_small.simple"):
    os.remove("C:/Users/z1325/Desktop/NLP_Torch/text-summarization/GitHub_T5/resources/datasets/wiki/wiki.test_small.simple")

file_write_obj = open("C:/Users/z1325/Desktop/NLP_Torch/text-summarization/GitHub_T5/resources/datasets/wiki/wiki.test_small.simple", 'w', encoding='utf-8')   # 新文件
for var in lst:
    file_write_obj.write(var) 

file_write_obj.close()
