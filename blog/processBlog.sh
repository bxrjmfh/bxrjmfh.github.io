python3 $BLOG_ROOT/blog/main.py
cd $BLOG_ROOT
git pull
git add .
echo  'enter the message'
git commit -m "A! from $IDENTIFIER_LH"
git push -u origin gh-pages-mytest

