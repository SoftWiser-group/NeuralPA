# ⚙️$`\mathbb{R}\mathrm{e}\mathbf{pilot}`$🛠️

<p align="left">
    <a href="https://arxiv.org/abs/2309.00608"><img src="https://img.shields.io/badge/arXiv-2309.00608-b31b1b.svg?style=for-the-badge">
    <a href="https://doi.org/10.5281/zenodo.8281250"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.8281250-blue?style=for-the-badge">
    <a href="https://hub.docker.com/r/universefly/repilot/tags"><img src="https://img.shields.io/badge/docker-universefly%2Frepilot-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white"></a>
</p>

ESEC/FSE'23 paper "Copiloting the Copilot: Fusing Large Language Models with Completion Engines for Automated Program Repair"

<img alt="Repilot Demo" src="assets/Repilot-Demo-Light.svg">

Repilot利用代码完成引擎和大型语言模型之间的协同作用，由大语言模型生成预测，再由language server剔除不可能的选项，以更有效地生成补丁。

作者表示只是以Automatic program repair来进行验证，此框架还可以很容易地应用于其他代码生成任务，包括代码完成、程序合成和生成测试。

#  build from Docker image

```bash
# 安装repilot镜像，大小约21G，注意预留好空间
# 容器内使用codeT5需要走代理，docker容器与主机在同一局域网内，注意需保证你使用的代理服务开启了局域网监听，容器内才能访问到代理服务，如果有虚拟网卡级代理则可忽略
# 第一次运行，安装
docker run -it --name repilot universefly/repilot:latest
# 后续运行使用，--add-host标志启用主机的默认名称 host.docker.internal。使用此标志启动容器以公开主机字符串
# -e标志更改容器内环境变量，注意此处host.docker.internal不能替换为127.0.0.1,后者指向的是容器环境的地址，前者指向的才是主机地址
# 将15777更改为你的代理使用的端口号。
docker run -it --add-host host.docker.internal:host-gateway -e https_proxy=http://host.docker.internal:15777 -e http_proxy=http://host.docker.internal:15777 universefly/repilot:latest
# 进入容器环境后可 curl google.com -v 检查是否成功

cd /root/Repilot

# 运行时环境
cat meta_config.json

# 使用CodeT5生成补丁，需要翻墙
# 展示生成的补丁以及通过language model判断的接受/拒绝
ACTIVE=1 python -m repilot.cli.main repair -b "Chart-9" --method pruned-mem -d chart-9-repilot -n 5

# 验证
python -m repilot.cli.main validate -d chart-9-repilot

# 展示图表
python -m repilot.cli.main evaluate -d chart-9-repilot
# You'll see something like this:
#                                              Repilot Evaluation Results                                              
# ┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
# ┃ Tag             ┃ Average Gen Time ┃ %Compilable Patches ┃ %Plausible Patches ┃ #Plausible Fixes ┃ #Correct Fixes ┃
# ┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
# │ chart-9-repilot │ 1.33s            │ 100.0%              │ 0.000%             │ 0                │ -              │
# └─────────────────┴──────────────────┴─────────────────────┴────────────────────┴──────────────────┴────────────────┘
```

For more details, check out [artifact documentation](/README-Artifact.md).

# build from source

> 参考[Dockerfile](https://github.com/ise-uiuc/Repilot/blob/main/Dockerfile)

> **Environment requirements**
> 
> - Python 3.10 and [Git LFS](https://git-lfs.com)
> - **Java 8, 11, 18**（全部需要）（管理多个java版本： [coursier](https://get-coursier.io/docs/cli-java)）
> - 运行CodeT5推荐GPU > 6G, Incoder-6.7B建议 > 30G.

### 配置Eclipse JDT Language Server

Follow the instructions in [the repo](https://github.com/UniverseFly/eclipse.jdt.ls) to build the modified Eclipse JDT Language Server. Note you will need Java 11:

```bash
git clone https://github.com/UniverseFly/eclipse.jdt.ls
cd eclipse.jdt.ls
JAVA_HOME=/path/to/java/11 ./mvnw clean verify -DskipTests=true
```

**Adjust** the following command according to your build to dry run the language server:

```bash
java \
	-Declipse.application=org.eclipse.jdt.ls.core.id1 \
	-Dosgi.bundles.defaultStartLevel=4 \
	-Declipse.product=org.eclipse.jdt.ls.core.product \
	-Dlog.level=ALL \
	-noverify \
	-Xmx1G \
	--add-modules=ALL-SYSTEM \
	--add-opens java.base/java.util=ALL-UNNAMED \
	--add-opens java.base/java.lang=ALL-UNNAMED \
	-jar ./plugins/org.eclipse.equinox.launcher_1.5.200.v20180922-1751.jar \
	-configuration ./config_linux \
	-data /path/to/data
```

### 安装Python包Repilot

```bash
git clone https://github.com/UniverseFly/Repilot && cd Repilot
# Do an editable install
pip install -e .
# Consider upgrading pip if you encounter any errors, also make sure you are using Python 3.10
# This command should also install all the dependencies of Repilot
```

### Defects4j数据集

Repilot evaluates on the [Defects4j](https://github.com/rjust/defects4j) dataset. Please checkout to its [v2.0.0 release](https://github.com/rjust/defects4j/releases/tag/v2.0.0) and follow its instructions to install the dataset.

> If you directly download the release instead of doing a checkout you may encounter errors when running Repilot, as Repilot will dump the metadata by collecting the meta information of these projects as Git repos. If they are not Git repos, Repilot may fail.

You can check the installation by running `/path/to/defects4j info -p Chart`.


### 准备运行时环境

We need to prepare a `meta_config.json` file for Repilot to work properly. The file should be placed in the root directory of Repilot. Please **modify** the following template according to your environment and save the file in the root directory of Repilot:

```json
{
  "d4j_home": "/home/yuxiang/Developer/defects4j",
  "d4j_checkout_root": "/home/yuxiang/Developer/d4j-checkout",
  "jdt_ls_repo": "/home/yuxiang/Developer/eclipse.jdt.ls",
  "java8_home": "/home/yuxiang/.cache/coursier/arc/https/github.com/AdoptOpenJDK/openjdk8-binaries/releases/download/jdk8u181-b13/OpenJDK8U-jdk_x64_linux_hotspot_8u181b13.tar.gz/jdk8u181-b13",
  "language_server_cmd": [
    "/home/yuxiang/.cache/coursier/arc/https/github.com/adoptium/temurin18-binaries/releases/download/jdk-18.0.2%252B9/OpenJDK18U-jdk_x64_linux_hotspot_18.0.2_9.tar.gz/jdk-18.0.2+9/bin/java",
    "-Declipse.application=org.eclipse.jdt.ls.core.id1",
    "-Dosgi.bundles.defaultStartLevel=4",
    "-Declipse.product=org.eclipse.jdt.ls.core.product",
    "-Dlog.level=ERROR",
    "-noverify",
    "-Xmx1G",
    "--add-modules=ALL-SYSTEM",
    "--add-opens",
    "java.base/java.util=ALL-UNNAMED",
    "--add-opens",
    "java.base/java.lang=ALL-UNNAMED",
    "-jar",
    "/home/yuxiang/Developer/eclipse.jdt.ls/org.eclipse.jdt.ls.product/target/repository/plugins/org.eclipse.equinox.launcher_1.6.400.v20210924-0641.jar",
    "-configuration",
    "/home/yuxiang/Developer/eclipse.jdt.ls/org.eclipse.jdt.ls.product/target/repository/config_linux"
  ],
  "seed": 0
}
```

Now let's `cd` back to the root directory of Repilot, and run the following command to checkout all the Defects4J bugs:

```bash
python -m repilot.cli.init
```

### 运行

```bash
# Generate patches with the full Repilot approach using CodeT5
ACTIVE=1 python -m repilot.cli.main repair -b "Chart-9" --method pruned-mem -d chart-9-repilot -n 5 # You will see logs about the patch generation and which tokens are accepted/rejected.

# Validate the patch generation
python -m repilot.cli.main validate -d chart-9-repilot

# Print a table of the evaluation results
python -m repilot.cli.main evaluate -d chart-9-repilot
```

