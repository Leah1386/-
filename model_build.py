import torch
import torch.nn as nn

# ===================== 全局设备配置 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===================== 1. 主模型：STA-LSTM =====================
class STA_LSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, output_size=1):
        super(STA_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 核心LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        # 全连接输出层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化隐层状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)

        # LSTM 前向传播
        out, _ = self.lstm(x, (h0, c0))
        # 取最后一个时间步输出
        out = self.fc(out[:, -1, :])
        return out.squeeze()


# ===================== 2. 对比模型：标准LSTM =====================
class LSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, output_size=1):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze()


# ===================== 3. 对比模型：双向LSTM (BiLSTM) =====================
class BiLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, output_size=1):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze()


# ===================== 4. 对比模型：GRU =====================
class GRU(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, output_size=1):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze()