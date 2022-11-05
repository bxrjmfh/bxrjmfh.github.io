python3 $BLOG_ROOT/blog/main.py

read -rsp $'Press any key to continue...\n' -n1 key
    cd $BLOG_ROOT
    git pull
    git add .
    echo  'enter the message'
    git commit -m "A! from $IDENTIFIER_LH"
    git push -u origin gh-pages-mytest



