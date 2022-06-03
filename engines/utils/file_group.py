import os


def _read_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read().replace('\n', '').replace('\t', '').replace('\u3000', '').replace('_!_', '')


def save_file(direname):
    f_train = open('../../data/cnews/cnews.train.txt', 'w', encoding='utf-8')
    f_test = open('../../data/cnews/cnews.test.txt', 'w', encoding='utf-8')
    f_val = open('../../data/cnews/cnews.val.txt', 'w', encoding='utf-8')

    for category in os.listdir(direname):
        print(category)
        cat_dir = os.path.join(direname, category)
        if not os.path.isdir(cat_dir):
            continue
        files = os.listdir(cat_dir)
        count = 0
        for cur_file in files:
            filename = os.path.join(cat_dir, cur_file)
            content = _read_file(filename)
            if count < 5000:
                f_train.write(category + '_!_' + content + '\n')
            elif count < 6000:
                f_test.write(category + '_!_' + content + '\n')
            else:
                f_val.write(category + '_!_' + content + '\n')
            count += 1

        print('finish:', category)

    f_train.close()
    f_val.close()
    f_test.close()


if __name__ == '__main__':
    new_dir = "D:\\yuqing\\cnews\\"
    save_file(new_dir)
    print(len(open('../../data/cnews/cnews.train.txt', 'r', encoding='utf-8').readlines()))
    print(len(open('../../data/cnews/cnews.test.txt', 'r', encoding='utf-8').readlines()))
    print(len(open('../../data/cnews/cnews.val.txt', 'r', encoding='utf-8').readlines()))
