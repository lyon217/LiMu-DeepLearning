import torch

# model = torch.load('./dataset/resnest50-528c19ca.pth')
# print(model)

# import torchvision.models as models
# models.resnet50

import timm

# 创建resnet34和efficientnet_b0模型
model = timm.create_model('resnet34', pretrained=True)
# model = timm.create_model('efficientnet_b0')

# 我们可以通过list_models函数来直接创建,有预训练参数的模型列表
all_pretrained_models_available = timm.list_models()
print(all_pretrained_models_available)
print(len(all_pretrained_models_available))
# 如果带参数pretrained=True是716  如果False是889

# create_model的函数还支持features_onlu=True参数
# ,此时函数将返回部分网络,该网络提取每一步最深一层的特征图,
# 还可以使用out_indices=[...]指定层的索引,以提起中间特征
x = torch.randn(1, 3, 224, 224)
model = timm.create_model('resnet34')
preds = model(x)
print(f'preds shape:{preds.shape}')
# preds shape:torch.Size([1, 1000])

all_feature_extractor = timm.create_model('resnet34', features_only=True)
all_features = all_feature_extractor(x)
print(f'All {len(all_features)} Features:')
for i in range(len(all_features)):
    print(f'feature{i} shape:{all_features[i].shape}')
# All 5 Features:
# feature0 shape:torch.Size([1, 64, 112, 112])
# feature1 shape:torch.Size([1, 64, 56, 56])
# feature2 shape:torch.Size([1, 128, 28, 28])
# feature3 shape:torch.Size([1, 256, 14, 14])
# feature4 shape:torch.Size([1, 512, 7, 7])

out_indices = [2, 3, 4, ]
selected_feature_extractor = timm.create_model('resnet34', features_only=True, out_indices=out_indices)
selected_features = selected_feature_extractor(x)
print("Selected Features:")
for i in range(len(out_indices)):
    print(f"feature{out_indices[i]} shape:{selected_features[i].shape}")
# Selected Features:
# feature2 shape:torch.Size([1, 128, 28, 28])
# feature3 shape:torch.Size([1, 256, 14, 14])
# feature4 shape:torch.Size([1, 512, 7, 7])

# 我们可以通过 timm_model及其features_only,out_indices参数将预训练模型方便地转换为自己想要的特征提取器。
feature_extractor = timm.create_model('resnet34', features_only=True, out_indices=[3])
print('type:', type(feature_extractor))
print('len: ', len(feature_extractor))
for item in feature_extractor:
    print(item)
# type: # <class 'timm.models.features.FeatureListNet'>
# len:  7
# conv1
# bn1
# act1
# maxpool
# layer1
# layer2
# layer3

# 可以看到，feature_extractor 其实也是一个神经网络，
# 在 timm 中称为 FeatureListNet，
# 而我们通过 out_indices 参数来指定截取到哪一层特征。


