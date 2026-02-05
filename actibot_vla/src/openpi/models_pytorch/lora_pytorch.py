import torch
from torch import nn

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, alpha=1.0, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 0.0  # 后面可以用根号r

        # 原始权重
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

        # LoRA 分支（r=0 时可以不创建）
        if r > 0:
            self.A = nn.Parameter(torch.zeros(r, in_features))
            self.B = nn.Parameter(torch.zeros(out_features, r))
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.A is not None:
            nn.init.zeros_(self.A)
            nn.init.zeros_(self.B)

    @classmethod
    def from_linear(cls, linear: nn.Linear, r: int, alpha: float):
        """从已有的 nn.Linear 创建一个带 LoRA 的线性层，并复制原始权重。"""
        lora_linear = cls(
            linear.in_features,
            linear.out_features,
            r=r,
            alpha=alpha,
            bias=linear.bias is not None,
        )
        # 拷贝原始权重
        with torch.no_grad():
            lora_linear.weight.copy_(linear.weight)
            if linear.bias is not None:
                lora_linear.bias.copy_(linear.bias)
        return lora_linear

    def forward(self, x):
        base = torch.nn.functional.linear(x, self.weight, self.bias)
        if self.A is None or self.r == 0:
            return base
        lora = (x @ self.A.t()) @ self.B.t() * self.scaling
        return base + lora

def test_lora_linear_r0():

    torch.manual_seed(0)
    x = torch.randn(4, 10)  # batch=4, in=10

    lin = nn.Linear(10, 6)
    lora_lin = LoRALinear.from_linear(lin, r=4, alpha=1.0)

    # 冻结 W 和 bias，只训练 A/B
    for name, p in lora_lin.named_parameters():
        if name in ["A", "B"]:
            p.requires_grad = True
        else:
            p.requires_grad = False

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, lora_lin.parameters()), lr=1e-3)

    for step in range(100):
        x = torch.randn(32, 10)
        target = torch.randn(32, 6)
        y = lora_lin(x)
        loss = ((y - target)**2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 20 == 0:
            print(step, loss.item())

def test_lora_linear_r4():
    torch.manual_seed(0)
    x = torch.randn(4, 10)  # batch=4, in=10

    # 普通 Linear
    lin = nn.Linear(10, 6)
    # 从这个 Linear 构造 LoRA Linear，r=8 -> 启用 LoRA 分支
    lora_lin = LoRALinear.from_linear(lin, r=4, alpha=1.0)

    # 初始时 A/B 被初始化为 0，结果应该一样
    y_base = lin(x)
    y_lora0 = lora_lin(x)
    print("init max diff:", (y_base - y_lora0).abs().max().item())  # ≈0

    # 手动改一下 LoRA 参数，看输出是否变化
    with torch.no_grad():
        lora_lin.A.normal_(std=0.1)
        lora_lin.B.normal_(std=0.1)

    y_lora1 = lora_lin(x)
    print("after LoRA change max diff:", (y_base - y_lora1).abs().max().item())  # >0



if __name__ == "__main__":
    # test_lora_linear_r0()
    test_lora_linear_r4()