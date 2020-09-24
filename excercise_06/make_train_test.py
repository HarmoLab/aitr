import csv

num_train = 3000
num_test = 500

header = ['filename', 'label']
with open('data/train.csv', 'w') as f:
    writer = csv.writer(f)
    start_no = 1000
    cat_list = [['cat.{}.jpg'.format(start_no + x), 0] for x in range(num_train)]
    dog_list = [['dog.{}.jpg'.format(start_no + x), 1] for x in range(num_train)]

    writer.writerow(header)
    writer.writerows(cat_list)
    writer.writerows(dog_list)


with open('data/test.csv', 'w') as f:
    writer = csv.writer(f)
    start_no = 10000
    cat_list = [['cat.{}.jpg'.format(start_no + x), 0] for x in range(num_test)]
    dog_list = [['dog.{}.jpg'.format(start_no + x), 1] for x in range(num_test)]

    writer.writerow(header)
    writer.writerows(cat_list)
    writer.writerows(dog_list)
