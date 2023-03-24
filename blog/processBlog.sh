cd $BLOG_ROOT
git pull
python3 $BLOG_ROOT/blog/main.py $RAW_BLOG_PATH $POST_FILES_PATH

read -rsp $'Press any key to continue...\n' -n1 key

git add .
echo  'enter the message'
git commit -m "A! from $IDENTIFIER_LH"
git push -u origin gh-pages-mytest



