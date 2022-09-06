参见[这里](https://stackoverflow.com/questions/48279878/how-to-access-python-return-value-from-bash-script)

python 的返回值是不能被shell直接读取的，因此需要使用stderr来传出数值。

具体的操作方法如下：
```python
import sys
    
    
def main():
    print ("exec main..")
    sys.stderr.write('execution ok\n')
    return "execution ok"
    
if __name__ == '__main__':
    main()
```

对应的shell脚本为:

```shell
#!/bin/bash
    
script_output=$(python foo.py 1>/dev/null)
echo $script_output
```

即可解决。
