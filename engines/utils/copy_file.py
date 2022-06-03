import os
import shutil
MAXCOUNT = 6500
source_dir = "G:\\数据集\\自然语言数据集\\中文\\新浪\\THUCNews\\THUCNews"
new_dir = "D:\\yuqing\\cnews\\"
categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
for category in os.listdir(source_dir):
    print(category)
    if category in categories:
        if not os.path.isdir(os.path.join(new_dir, category)):
            os.mkdir(os.path.join(new_dir, category))
        source_file_list = os.listdir(os.path.join(source_dir, category))
        first_file_num = int(source_file_list[0].split('.')[0])
        for i in range(first_file_num, first_file_num + MAXCOUNT):
            source_category_path = os.path.join(source_dir, category)
            target_category_path = os.path.join(new_dir, category)
            new_file_name = str(i) + '.txt'
            try:
                shutil.copy(os.path.join(source_category_path, new_file_name), os.path.join(target_category_path, new_file_name))
            except:
                print("Unable to copy file")
