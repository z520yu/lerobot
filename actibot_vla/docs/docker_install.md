# OpenPI Docker环境完整安装指南


## ⚠️ 重要要求

根据项目要求，必须满足以下条件：
- ✅ 使用rootless模式安装Docker
- ❌ 不能使用snap版本的Docker  
- ❌ 不能使用Docker Desktop
- ✅ 需要NVIDIA Container Toolkit支持GPU

## 🚀 快速安装（Ubuntu 22.04）

项目提供了便捷脚本，可以一键完成安装：

```bash
# 安装Docker（rootless模式）
bash scripts/docker/install_docker_ubuntu22.sh

# 安装NVIDIA Container Toolkit
bash scripts/docker/install_nvidia_container_toolkit.sh

# 重启以确保配置生效
sudo reboot
```

## 📋 手动安装步骤

### 第一步：清理现有Docker安装

```bash
# 停止Docker服务
sudo systemctl stop docker containerd

# 卸载snap版本（项目要求）
sudo snap remove docker

# 卸载Docker Desktop（项目要求）
sudo apt remove docker-desktop

# 卸载其他Docker版本
sudo apt-get remove docker docker-engine docker.io containerd runc docker-ce docker-ce-cli

# 清理残留文件
sudo rm -rf /var/lib/docker /var/lib/containerd
```

### 第二步：安装系统依赖

```bash
# 更新包列表
sudo apt-get update

# 安装必要依赖
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    apt-transport-https \
    software-properties-common \
    uidmap \
    dbus-user-session \
    fuse-overlayfs
```

### 第三步：添加Docker官方仓库

```bash
# 创建密钥目录
sudo install -m 0755 -d /etc/apt/keyrings

# 添加Docker GPG密钥
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# 添加Docker仓库
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 更新包列表
sudo apt-get update
```

### 第四步：安装Docker Engine

```bash
# 安装最新版本
sudo apt-get install -y \
    docker-ce \
    docker-ce-cli \
    containerd.io \
    docker-buildx-plugin \
    docker-compose-plugin

# 验证安装
docker --version
docker compose version
```

### 第五步：配置Rootless模式（关键）

```bash
# 添加用户到docker组
sudo usermod -aG docker $USER

# 安装rootless Docker
curl -fsSL https://get.docker.com/rootless | sh

# 配置环境变量
echo 'export PATH=/home/'$USER'/bin:$PATH' >> ~/.bashrc
echo 'export DOCKER_HOST=unix:///run/user/'$(id -u)'/docker.sock' >> ~/.bashrc

# 重新加载环境
source ~/.bashrc

# 启用用户级服务
systemctl --user enable docker
systemctl --user start docker

# 设置开机自启
sudo loginctl enable-linger $USER
```

### 第六步：安装NVIDIA Container Toolkit

如果您有NVIDIA GPU：

```bash
# 添加NVIDIA仓库
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 安装Container Toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 配置Docker运行时
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 配置rootless模式的NVIDIA支持
nvidia-ctk runtime configure --runtime=docker --config=$HOME/.config/docker/daemon.json
systemctl --user restart docker
```

## ✅ 验证安装

### 基本功能验证

```bash
# 检查Docker版本
docker --version
docker compose version

# 测试基本功能
docker run --rm hello-world

# 检查rootless模式
docker info | grep -i "docker root dir"
# 应显示用户目录路径，而非 /var/lib/docker

# 检查用户服务状态
systemctl --user status docker
```

### GPU支持验证

```bash
# 测试GPU访问
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi

# 如果成功，应显示GPU信息
```

## 🐳 运行OpenPI容器

### 构建和启动主容器

```bash
# 进入项目目录
cd /path/to/openpi

# 构建并运行主容器
docker compose -f scripts/docker/compose.yml up --build

# 后台运行
docker compose -f scripts/docker/compose.yml up --build -d
```

### 运行特定示例

```bash
# ALOHA仿真示例
docker compose -f examples/aloha_sim/compose.yml up --build

# DROID示例
docker compose -f examples/droid/compose.yml up --build

# LIBERO示例
docker compose -f examples/libero/compose.yml up --build
```

### 进入容器进行交互

```bash
# 进入运行中的容器
docker compose -f scripts/docker/compose.yml exec openpi bash

# 或启动新的交互式容器
docker run -it --gpus all openpi:latest bash
```

## ⚙️ 容器配置

### 环境变量设置

```bash
# 设置数据目录
export OPENPI_DATA_HOME=~/.cache/openpi

# GPU内存优化
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

### 数据卷挂载

```yaml
# docker-compose.yml 示例
services:
  openpi:
    volumes:
      - ~/.cache/openpi:/root/.cache/openpi
      - ./checkpoints:/workspace/checkpoints
      - ./dataset:/workspace/dataset
```

## 🔧 故障排除

### 常见问题解决

**问题1：权限被拒绝**
```bash
# 确保用户在docker组中
groups $USER
# 如果没有，重新登录或运行：
newgrp docker
```

**问题2：GPU不可用**
```bash
# 检查NVIDIA驱动
nvidia-smi

# 检查Container Toolkit
nvidia-ctk --version

# 重新配置运行时
sudo nvidia-ctk runtime configure --runtime=docker
systemctl --user restart docker
```

**问题3：容器构建失败**
```bash
# 清理Docker缓存
docker system prune -a

# 重新构建，不使用缓存
docker compose build --no-cache
```

**问题4：网络问题**
```bash
# 检查Docker网络
docker network ls

# 重置网络配置
docker network prune
```

### 调试命令

```bash
# 查看容器日志
docker compose logs openpi

# 检查容器状态
docker compose ps

# 进入容器调试
docker compose exec openpi bash

# 检查GPU在容器内的可用性
docker compose exec openpi nvidia-smi
```

## 🚀 性能优化

### 容器资源限制

```yaml
# docker-compose.yml
services:
  openpi:
    deploy:
      resources:
        limits:
          memory: 32G
          cpus: '8'
        reservations:
          memory: 16G
          cpus: '4'
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### 存储优化

```bash
# 使用本地SSD挂载数据目录
# 避免频繁的容器-主机文件传输
# 使用.dockerignore减少构建上下文
```

## 📚 高级用法

### 多容器编排

```yaml
# 完整的多服务配置示例
services:
  openpi-train:
    build: .
    volumes:
      - ./checkpoints:/workspace/checkpoints
    environment:
      - XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    
  openpi-serve:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      - openpi-train
    command: ["scripts/serve_policy.py", "policy:checkpoint"]
```

### 开发环境配置

```bash
# 开发模式挂载源代码
docker run -it --gpus all \
    -v $(pwd):/workspace \
    -v ~/.cache/openpi:/root/.cache/openpi \
    openpi:latest bash
```
