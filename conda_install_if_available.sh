while read requirement
do conda install --yes $requirement || pip3 install $requirement
done < requirements_spvit.txt


# 可以直接把环境文件夹复制到新服务器的conda 环境路径下面！！！