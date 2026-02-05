echo "🚀 开始安装Docker（改进版）..."

# 清理旧配置
sudo rm -f /etc/apt/sources.list.d/docker.list
sudo rm -f /etc/apt/keyrings/docker.asc
sudo rm -f /etc/apt/keyrings/docker.gpg

# 安装依赖
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg lsb-release

# 创建密钥目录
sudo mkdir -p /etc/apt/keyrings

# 尝试多种方式添加GPG密钥
echo "📦 添加Docker GPG密钥..."
if curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg; then
    echo "✅ 官方源GPG密钥添加成功"
    DOCKER_REPO="https://download.docker.com/linux/ubuntu"
elif curl -fsSL https://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg; then
    echo "✅ 阿里云镜像GPG密钥添加成功"
    DOCKER_REPO="https://mirrors.aliyun.com/docker-ce/linux/ubuntu"
else
    echo "❌ GPG密钥添加失败，请检查网络连接"
    exit 1
fi

# 添加仓库
echo "📦 添加Docker仓库..."
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] $DOCKER_REPO \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 更新包列表
sudo apt-get update

# 安装Docker
echo "📦 安装Docker组件..."
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 配置用户权限
echo "⚙️ 配置用户权限..."
sudo usermod -aG docker $USER

# 启用服务
echo "🔧 启用Docker服务..."
sudo systemctl enable docker.service
sudo systemctl enable containerd.service
sudo systemctl start docker.service

# 验证安装
echo "✅ 验证安装..."
if docker --version; then
    echo "🎉 Docker安装成功！"
else
    echo "❌ Docker安装可能存在问题"
fi

echo ""
echo "********************************************************************"
echo "**** 请重新登录或运行 'newgrp docker' 以使权限更改生效 ****"
echo "********************************************************************"
echo ""
