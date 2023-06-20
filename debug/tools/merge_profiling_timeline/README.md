# Profiling merge tool

## 介绍
本工具支持合并profiling的timeline数据，支持合并指定rank的timline、合并指定timeline中的item

## 软件架构
软件架构说明

## 使用说明
> 合并单机多卡timeline时，不涉及多机间时间校准，直接忽略第1节内容
### 1 获取服务器与客户端的启动时间差值文件time_difference.json

   以下代码中，服务器为当前代码所在机器，所有步骤都在服务器上操作，可以选择集群中任意节点作为服务器

#### 1.1 拉取代码到集群任意节点任意目录，进入get_nodes_timediff目录

   ```shell
   git clone https://gitee.com/aerfaliang/merge_profiling_timeline.git
   cd merge_profiling_timeline/get_nodes_timediff
   ```

#### 1.2 安装依赖sshpass(1.4中远程执行客户端命令时要用到)

##### Ubuntu

   ```shell
   apt-get install sshpass
   ```

##### CentOS

   ```shell
    # 源码包安装
    wget http://sourceforge.net/projects/sshpass/files/sshpass/1.05/sshpass-1.05.tar.gz 
    tar -xvzf sshpass-1.05.tar.gz 
    cd sshpass-1.05
    ./configure 
    make 
    make install 

    # yum安装
    yum -y install sshpass
   ```

#### 1.3 按照集群节点顺序编辑nodeinfo.json文件  

   文件中的内容为全部节点的ip，用户名，密码以及端口

   注意：必须按照集群节点顺序填写，以下为样例文件中内容，请根据实际节点信息修改，将集群中所有节点信息填入

   ```shell
   root@root:~/test/merge_profiling_timeline/get_nodes_timediff$ cat nodeinfo.json
   {
       "cluster": {
           "90.90.66.62": {
               "user": "dcs-62",
               "pd": "password",
               "port": 22
           },
           "90.90.66.64": {
               "user": "dcs-64",
               "pd": "password",
               "port": 22  
               
           }
       }
   }
   
   ```

#### 1.4 执行get_nodes_timediff.sh脚本，获取服务器与客户端的启动时间差值文件time_difference.json

   ```shell
   # bash get_nodes_timediff.sh {当前机器ip}
   bash get_nodes_timediff.sh 90.90.66.62
   ```

#### 1.5 检查在脚本同目录有time_difference.json文件生成

   文件中记录的是节点ip以及客户端相对服务器启动时间差(客户端当前启动时间-服务器启动时间)，按照集群节点顺序排列

   ```shell
   root@root:~/test/merge_profiling_timeline/get_nodes_timediff$ cat time_difference.json
   {
       "90.90.66.62": -3.8049183785915375e-06,
       "90.90.66.64": -1.551163767464459
   }
   ```

### 2 多timeline数据合并

#### 2.1 使用msporf采集数据，将采集到的所有节点的profiling数据拷贝到当前机器同一目录下，以下假设数据在/home/test/all_cann_profiling下

#### 2.2 使用merge_profiling_timeline下的main.py合并timeline

可选参数：
- -d: **必选参数**，profiling数据文件或文件夹路径
- -t: **当需要融合节点间timeline时为必选参数**，传入的为前面步骤1.3生成的time_difference.json文件路径
- -o: 可选参数，指定合并后的timeline文件输出的路径，默认为'-d'输入的路径
- --rank：可选参数，指定需要合并timeline的卡号，默认全部合并
- --items：可选参数，指定需要合并的profiling数据项，默认全部合并

profiling数据目录结构示意, 合并timeline必需数据：`msprof.json`和`info.json.*`：
```
|- cann_profiling
    |- PROF_***
        |- timeline
            |- msprof.json
        |- device_*
            |- info.json.*
        ...
    |- PROF_***
    ...
```


**使用示例**：

1、合并单机多卡timeline，默认合并所有卡、所有数据项：
```
python3 main.py -d path/to/cann_profiling/
```

2、合并单机多卡timeline，只合并0卡和1卡：

```
python3 main.py -d path/to/cann_profiling/ --rank 0,1
```

3、合并单机多卡timeline，合并所有卡的CANN层和Ascend_Hardware层数据
```
python3 main.py -d path/to/cann_profiling/ --items CANN,Ascend_Hardware
```

4、合并多机多卡的timeline时, 需要-t指定节点间的时间误差文件路径, 用以校准节点间的时间：


```
python3 main.py -d path/to/cann_profiling/ -t path/to/time_difference.json --rank 0,8,
```

合并timeline查看：
> 查看在 -o 指定的目录（默认在-d指定的目录下）生辰过的msprof_merged_*p.json，为合并后的文件


### 3、超大timeline文件查看

直接使用以下命令
```
cd merge_profiling_timeline
python ./trace_processor --httpd path/to/msprof_merged_*p.json 
```
等待加载完毕，刷新[perfetto](https://ui.perfetto.dev/)界面，点击`YES, use loaded trace`即可展示timeline




