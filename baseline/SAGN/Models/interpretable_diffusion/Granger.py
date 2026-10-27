# import torch
# from torch import nn
# import torch.nn.functional as F

# class CausalCrossAttention(nn.Module):
#     def __init__(self, num_features, b_dim, head_dim=64, num_heads=4):
#         super().__init__()
#         self.num_features = num_features
#         self.b_dim = b_dim  # B的维度
#         self.head_dim = head_dim
#         self.num_heads = num_heads
        
#         # A作为Query - 保持不变
#         self.a_to_query = nn.Linear(num_features, num_heads * head_dim)
#         # B作为Key和Value - 输入维度改为b_dim
#         self.b_to_key = nn.Linear(b_dim, num_heads * head_dim)
#         self.b_to_value = nn.Linear(b_dim, num_heads * head_dim)
        
#         # 输出变换
#         self.output_proj = nn.Linear(num_heads * head_dim, num_features)

#         # 门控权重生成层 - 输出单个标量
#         self.gate = nn.Linear(num_features, 1)
        
#     def forward(self, A, B):
#         # A: (B, W, F), B: (B, W, 2)   
#         batch_size, seq_len, _ = A.shape
        
#         # 生成Query, Key, Value
#         Q = self.a_to_query(A).view(batch_size, seq_len, self.num_heads, self.head_dim)
#         K = self.b_to_key(B).view(batch_size, seq_len, self.num_heads, self.head_dim)  
#         V = self.b_to_value(B).view(batch_size, seq_len, self.num_heads, self.head_dim)  
        
#         # 转换维度
#         Q = Q.transpose(1, 2)  # (B, num_heads, W, head_dim)
#         K = K.transpose(1, 2)
#         V = V.transpose(1, 2)
        
#         # 计算注意力分数
#         attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
#         # 应用因果掩码
#         causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
#         causal_mask = causal_mask.to(attention_scores.device)
#         attention_scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
#         attention_weights = torch.softmax(attention_scores, dim=-1)
        
#         # 应用注意力
#         enhanced_features = torch.matmul(attention_weights, V)
#         enhanced_features = enhanced_features.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
#         # 残差连接
#         output = self.output_proj(enhanced_features)

#         gate_weight = self.gate(output.mean(dim=1, keepdim=True))  # (B, 1, 1)
#         gate_weight = torch.sigmoid(gate_weight)  # 范围[0, 1]
#         return A+output, gate_weight
    
# class GrangerNet(nn.Module):
#     def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.1, prediction_steps=5, seq_len=20):
#         super().__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.prediction_steps = prediction_steps
#         self.his = seq_len - prediction_steps
        
#         # 编码器
#         self.encoder = nn.GRU(input_dim, hidden_dim, num_layers, 
#                              batch_first=True, dropout=dropout)
        
#         # 注意力机制聚合历史信息
#         self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
#         # 受限模型
#         self.restricted_predictor = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim // 2, prediction_steps * input_dim)
#         )
        
#         # 完整模型
#         self.full_predictor = nn.Sequential(
#             nn.Linear(self.his*input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, prediction_steps * input_dim)
#         )
        
#         # 因果性网络
#         self.causality_net = nn.Sequential(
#             nn.Linear(hidden_dim + input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, input_dim),
#             nn.Sigmoid()
#         )

#         self.cross_attention = CausalCrossAttention(num_features=input_dim,b_dim=2)
        
#     def forward(self, A, B):
#         batch_size, seq_len, feature_dim = A.shape
#         hist_len = seq_len - self.prediction_steps
        
#         A_hist = A[:, :hist_len, :]
#         A_future = A[:, hist_len:, :]
#         B_hist = B[:, :hist_len, :]
        
#         # 编码历史信息
#         hist_features, _ = self.encoder(A_hist)  # (B, hist_len, hidden_dim)
        
#         # 使用注意力聚合历史信息
#         aggregated_hist, _ = self.attention(
#             hist_features, hist_features, hist_features
#         )  # (B, hist_len, hidden_dim)
#         last_hist = torch.mean(aggregated_hist,dim=1)  # (B, hidden_dim)

#         # 受限模型预测：仅使用A的历史
#         restricted_pred = self.restricted_predictor(last_hist)  # (B, prediction_steps * F)
#         restricted_pred = restricted_pred.reshape(batch_size, self.prediction_steps, feature_dim)  # (B, 5, F)
        
#         # 完整模型预测：使用A和B的历史
#         B_gredient = torch.diff(B_hist, dim=1)  # 在第二个维度(序列维度)求差分
#         rps_raw_grad = torch.cat([torch.zeros(B_hist.shape[0], 1, B_hist.shape[2]).cuda(), B_gredient], dim=1)
#         rps_data = torch.cat([B_hist, rps_raw_grad], dim=-1)
#         full_input, _ = self.cross_attention(A_hist,rps_data)
#         full_pred = self.full_predictor(full_input.reshape(batch_size,-1))  # (B, prediction_steps * F)
#         full_pred = full_pred.reshape(batch_size, self.prediction_steps, feature_dim)  # (B, 5, F)
        
#         # 计算预测误差（沿时间维度平均）
#         restricted_error = F.mse_loss(restricted_pred, A_future, reduction='none').mean(dim=1)  # (B, F)
#         full_error = F.mse_loss(full_pred, A_future, reduction='none').mean(dim=1)  # (B, F)
        
#         # 计算Granger因果性强度
#         error_reduction = torch.clamp(restricted_error - full_error, min=0)  # Captures directional influence from B to A
#         causality_strength = error_reduction / (restricted_error + 1e-8)  # Using ratio makes it comparable across different scales
        
#         # # 使用神经网络进一步提炼因果性强度
#         # causality_input = torch.cat([last_hist, causality_strength], dim=-1)  # (B, hidden_dim + F)
#         # refined_causality = self.causality_net(causality_input)  # (B, F)
        
#         return torch.mean(restricted_error), causality_strength


import torch
from torch import nn
import torch.nn.functional as F

class CausalCrossAttention(nn.Module):
    def __init__(self, num_features, b_dim, head_dim=64, num_heads=4):
        super().__init__()
        self.num_features = num_features
        self.b_dim = b_dim  # B的维度
        self.head_dim = head_dim
        self.num_heads = num_heads
        
        # A作为Query - 保持不变
        self.a_to_query = nn.Linear(num_features, num_heads * head_dim)
        # B作为Key和Value - 输入维度改为b_dim
        self.b_to_key = nn.Linear(b_dim, num_heads * head_dim)
        self.b_to_value = nn.Linear(b_dim, num_heads * head_dim)
        
        # 输出变换
        self.output_proj = nn.Linear(num_heads * head_dim, num_features)

        # 门控权重生成层 - 输出单个标量
        self.gate = nn.Linear(num_features, 1)
        
    def forward(self, A, B):
        # A: (B, W, F), B: (B, W, 2)   
        batch_size, seq_len, _ = A.shape
        
        # 生成Query, Key, Value
        Q = self.a_to_query(A).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.b_to_key(B).view(batch_size, seq_len, self.num_heads, self.head_dim)  
        V = self.b_to_value(B).view(batch_size, seq_len, self.num_heads, self.head_dim)  
        
        # 转换维度
        Q = Q.transpose(1, 2)  # (B, num_heads, W, head_dim)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # 应用因果掩码
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.to(attention_scores.device)
        attention_scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        # 应用注意力
        enhanced_features = torch.matmul(attention_weights, V)
        enhanced_features = enhanced_features.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # 残差连接
        output = self.output_proj(enhanced_features)

        gate_weight = self.gate(output.mean(dim=1, keepdim=True))  # (B, 1, 1)
        gate_weight = torch.sigmoid(gate_weight)  # 范围[0, 1]
        return A+output, gate_weight
    
class GrangerNet(nn.Module):
    def __init__(self, input_dim, his_len, hidden_dim=128, num_layers=2, dropout=0.1, prediction_steps=5,
                 joint_training=False):
        super().__init__()
        self.joint_training = joint_training
        self.last_full_error = None
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.prediction_steps = prediction_steps
        self.his = his_len - prediction_steps
        
        # 编码器
        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers, 
                             batch_first=True, dropout=dropout)
        self.linear = nn.Linear(self.his, hidden_dim)
        # 注意力机制聚合历史信息
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # 受限模型
        self.restricted_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, prediction_steps)
        )
        
        # 完整模型
        self.full_predictor = nn.Sequential(
            nn.Linear(self.his, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, prediction_steps)
        )
        
        # 因果性网络
        self.causality_net = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

        self.cross_attention = CausalCrossAttention(num_features=input_dim,b_dim=2)
        
    def forward(self, A, B):
        batch_size, seq_len, feature_dim = A.shape
        hist_len = seq_len - self.prediction_steps
        
        A_hist = A[:, :hist_len, :]
        A_future = A[:, hist_len:, :]
        B_hist = B[:, :hist_len, :]
        
        # 编码历史信息
        #hist_features, _ = self.encoder(A_hist)  # (B, hist_len, hidden_dim)
        hist_features = self.linear(A_hist.transpose(1, 2))
        # 使用注意力聚合历史信息
        aggregated_hist, _ = self.attention(
            hist_features, hist_features, hist_features
        )  # (B, feature_dim, hidden_dim)
        #last_hist = torch.mean(aggregated_hist,dim=1)  # (B, hidden_dim)

        # 受限模型预测：仅使用A的历史
        restricted_pred = self.restricted_predictor(aggregated_hist)  # (B, prediction_steps * F)
        restricted_pred = restricted_pred.transpose(1, 2)
        
        # 完整模型预测：使用A和B的历史
        B_gredient = torch.diff(B_hist, dim=1)  # 在第二个维度(序列维度)求差分
        rps_raw_grad = torch.cat([torch.zeros_like(B_hist[:, :1]), B_gredient], dim=1)
        rps_data = torch.cat([B_hist, rps_raw_grad], dim=-1)
        full_input, load_gate = self.cross_attention(A_hist,rps_data)
        if self.joint_training:
            full_input = A_hist + load_gate * (full_input - A_hist)
        full_pred = self.full_predictor(full_input.transpose(1, 2))  # (B, prediction_steps * F)
        full_pred = full_pred.transpose(1, 2)
        
        # 计算预测误差（沿时间维度平均）
        restricted_error = F.mse_loss(restricted_pred, A_future, reduction='none').mean(dim=1)  # (B, F)
        full_error = F.mse_loss(full_pred, A_future, reduction='none').mean(dim=1)  # (B, F)
        self.last_full_error = full_error.detach()
        
        # 计算Granger因果性强度
        error_reduction = torch.clamp(restricted_error - full_error, min=0)  # Captures directional influence from B to A
        causality_strength = error_reduction / (restricted_error + 1e-8)  # Using ratio makes it comparable across different scales
        
        # # 使用神经网络进一步提炼因果性强度
        # causality_input = torch.cat([last_hist, causality_strength], dim=-1)  # (B, hidden_dim + F)
        # refined_causality = self.causality_net(causality_input)  # (B, F)
        
        if self.joint_training:
            return (restricted_error.mean() + full_error.mean()) * 0.5, causality_strength
        return torch.mean(restricted_error), causality_strength
