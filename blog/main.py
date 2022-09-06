# This is a script for preprocess the markdown blog file
# including function below :
#   -edit the title that can match jekyll post format , include date
#   -detect the  YAML front matter , if not ,add one and hit for fill necessary
#   -mv files to selected post dir and detect inplace files

import os
import datetime as dt
import pickle
import sys

RAW_FILE_PATH = './Documents'
POST_FILE_PATH = '/Data/Rice_Bowl/BLOG/_posts'
INFO_PATH = os.path.join(POST_FILE_PATH, 'fileInfos.pkl')
CATE_INFO_PATH = os.path.join(POST_FILE_PATH,'cateInfos.pkl')
TAG_INFO_PATH = os.path.join(POST_FILE_PATH,'tagInfos.pkl')

try:
    infile = open(INFO_PATH, 'rb')
    fileInfos = pickle.load(infile)
except:
    fileInfos = dict()
#     load fileInfos
try:
    infile = open(CATE_INFO_PATH,'rb')
    cateInfos = pickle.load(infile)
except:
    cateInfos = []

try:
    infile = open(TAG_INFO_PATH,'rb')
    tagInfos = pickle.load(infile)
except:
    tagInfos = []

newFiles = []
for root, dirs, files in os.walk(RAW_FILE_PATH):
    for i, filename in enumerate(files):
        print("processing the {} th files : ".format(i + 1) + filename)
        fileDir = os.path.join(RAW_FILE_PATH, filename)
        modifyTime = dt.datetime.utcfromtimestamp(os.path.getmtime(fileDir)).strftime("%Y-%m-%d")
        print("modified on "+modifyTime)
        newName = modifyTime + '-' + filename.replace(' ', '-')
        # edit the filename
        newFileDir = os.path.join(POST_FILE_PATH, newName)
        # shutil.copy(fileDir, newFileDir)
        if filename not in fileInfos:
            # not included in doc .
            # move the file
            with open(fileDir, "r") as readObj, open(newFileDir, "w+") as writeObj:
                print("Moved to new path : " + newFileDir)
                lines = [line for line in readObj]
                if "---" not in lines[0]:
                    writeObj.write("---\n")
                    # write the head
                    writeObj.write("layout: post\n")
                    writeObj.write('title: ' + filename.replace('.md', '') + "\n")
                    for i, l in enumerate(lines):
                        print(l)
                        if i > 20:
                            break
                    print("please input categories in this article , using blank to split :")
                    print(cateInfos)
                    categories = input()
                    cateInfos+=[c for c in categories.split() if c not in cateInfos]
                    writeObj.write("categories: " + categories + "\n")
                    print("please input tags in this article , using blank to split :")
                    print(tagInfos)
                    tags = input()
                    tagInfos+=[t for t in tags.split() if t not in tagInfos]
                    writeObj.write("tags: " + tags + "\n" + "---\n")
                    for l in lines:
                        writeObj.write(l)
                    writeObj.close()
                    print("finished !")
                    fileInfo = {filename:modifyTime}
                    fileInfos.update(fileInfo)
                    newFiles.append(filename)
        elif fileInfos[filename] != modifyTime:
            YAML = []
            with open(newFileDir, "r") as readObjHead:
                headCounter = 0
                for l in readObjHead:
                    if "---" in l:
                        headCounter += 1
                    YAML.append(l)
                    if headCounter == 2:
                        break
            with open(fileDir, "r") as readObj:
                lines = [l for l in readObj]
            with open(newFileDir, "w+") as writeObj:
                for l in YAML:
                    writeObj.write(l)
                for l in lines:
                    writeObj.write(l)
            print("Update the modify"+filename)
            fileInfo[filename] = modifyTime
            newFiles.append(filename)
with open(INFO_PATH, 'wb') as outfile:
    pickle.dump(fileInfo, outfile, pickle.HIGHEST_PROTOCOL)
    print("finished fileinfo saving")
with open(CATE_INFO_PATH, 'wb') as outfile:
    pickle.dump(cateInfos, outfile, pickle.HIGHEST_PROTOCOL)
    print("finished cateinfo saving")
with open(TAG_INFO_PATH, 'wb') as outfile:
    pickle.dump(tagInfos, outfile, pickle.HIGHEST_PROTOCOL)
    print("finished taginfo saving")
sys.stderr.write('Updated {} file(s) \n'.format(len(newFiles)+1))
for i in new files:
    sys.stderr.write(i+"\n")

