
from github_public_code.pvtv2 import pvt_v2_b2
from github_public_code.layer_modules import *

class simple_Transformer_Module_3_8(nn.Module):
    def __init__(self, channel=32):
        super(simple_Transformer_Module_3_8, self).__init__()

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = r'E:\py_project\medical_image_segmentation\polyp Segmenattion\pretrained_weight/pvt_v2_b2.pth'

        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.Translayer2_0 = BasicConv2d(64, channel, 1)
        self.Translayer2_1 = BasicConv2d(128, channel, 1)
        self.Translayer3_1 = BasicConv2d(320, channel, 1)
        self.Translayer4_1 = BasicConv2d(512, channel, 1)

        self.Level_Cat = Level_Cat_3_8(channel=channel)

        self.out_1 = nn.Conv2d(channel, 1, 1)
        self.out_2 = nn.Conv2d(channel * 2, 1, 1)
        self.out_3 = nn.Conv2d(channel * 3, 1, 1)
        self.out_4 = nn.Conv2d(channel * 4, 1, 1)
        self.out = nn.Conv2d(channel, 1, 1)

    def forward(self, x):
        # backbone
        pvt = self.backbone(x)
        x1 = pvt[0]  # [b, 64, 88, 88]
        x2 = pvt[1]  # [b, 128, 44, 44]
        x3 = pvt[2]  # [b, 320, 22, 22]
        x4 = pvt[3]  # [b, 512, 11, 11]

        x1_t = self.Translayer2_0(x1)  # [b, 32, 88, 88]
        x2_t = self.Translayer2_1(x2)  # [b, 32, 44, 44]
        x3_t = self.Translayer3_1(x3)  # [b, 32, 22 ,22]
        x4_t = self.Translayer4_1(x4)  # [b, 32, 11 ,11]
        # x1_out, x2_out, x3_out, cfm_feature = self.Level_Cat(x4_t, x3_t, x2_t)  # [b, 32, 11, 11], [b, 64, 22, 22], [b, 96, 44, 44], [b, 32, 44, 44]

        #x1_out, x2_2_out, x3_2_out, x4_2_out, out = self.Level_Cat_All(x4_t, x3_t, x2_t, T2)
        x1_out, x3_out, out = self.Level_Cat(x4_t, x3_t, x2_t, x1_t)
        prediction1 = self.out_1(x1_out)
        #prediction2 = self.out_2(x2_2_out)
        prediction3 = self.out_3(x3_out)
        #prediction4 = self.out_4(x4_2_out)
        prediction5 = self.out(out)

        prediction1 = F.interpolate(prediction1, scale_factor=32, mode='bilinear')
        prediction3 = F.interpolate(prediction3, scale_factor=8, mode='bilinear')
        prediction5 = F.interpolate(prediction5, scale_factor=4, mode='bilinear')

        return prediction1, prediction3, prediction5