cd $BLOG_ROOT
git pull
python3 $BLOG_ROOT/blog/main.py $RAW_BLOG_PATH $POST_FILES_PATH

#read -p $'Press any key to continue...\n' -n1 key

printf >&2 '%s ' 'continue ? (y/n)'
read ans
git add .
echo  'enter the message'
git commit -m "A! from $IDENTIFIER_LH"
git push -u origin gh-pages-mytest



