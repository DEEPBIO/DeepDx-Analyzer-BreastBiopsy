#!/usr/bin/env python3

import torch.utils.data
import torchvision.transforms as transforms
from Analyzer.exceptions import MagnificationError

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 patch_len=512,
                 slide=None,
                 xy_coord=None,
                 slide_level_index=[1, 2],
                 scale=1,
                 downsamples = [0, 0],
                 raw_mpp_x = 0.25):

        self.xy_coord = xy_coord
        self.patch_len = patch_len
        self.slide = slide
        self.scale = scale
        self.modifier = int((self.patch_len * 3 // 2) * scale)
        self.down_level = 1
        self.slide_level_index = slide_level_index
        self.downsamples = downsamples
        self.raw_mpp_x = raw_mpp_x

    def __len__(self):
        return len(self.xy_coord)

    def __getitem__(self, index):
        """
            coord는 slide_level이 0인 상태에서 patch를 뽑을 location의 좌표값을 의미합니다.
            high_res_patch는 slide_level[0], mpp=1 기준으로 1024 사이즈로 patch를 의미합니다.
            즉, mpp2 기준으로 512x512 pixel의 patch를 뽑는다는 의미입니다.
            slide_level이 존재할 경우에는, scaling이 필요가 없지만, slide_level의 값이 0인 경우에는, mpp를 1로 맞춰줘야 합니다.
            그러므로, slide_level의 값이 0인 경우에는 512*2*scale = 1024*scale을 곱해주게 됩니다.
            low_res_patch의 경우에는 mpp를 8로 맞춰줘야 하므로, 여기서 4를 더 곱해주게 됩니다.
        """

        coord = self.xy_coord[index]
        # mpp2 이상을 지원하는 wsi인 경우
        # mpp2 / 현재 mpp, 이 값을 patch(512)에 곱해주면 현재 mpp에서 추출해야 할 patch의 길이를 구할 수 있습니다.
        if self.slide_level_index[0] != 0:
            high_ratio = 2 / self.raw_mpp_x / self.downsamples[0]
            high_res_patch = self.slide.read_region((coord[0], coord[1]), self.slide_level_index[0],
                                                    (int(self.patch_len * high_ratio), int(self.patch_len * high_ratio)))
        # mpp2 이상을 지원하지 않는 경우,
        # slide_level 0에서, mpp2 기준으로 이미지 패치를 추출합니다. 시간이 매우 오래 걸리게 됩니다.
        elif self.slide_level_index[0] == 0:
            high_res_patch = self.slide.read_region((coord[0], coord[1]), self.slide_level_index[0],
                                                    (int(self.patch_len*self.scale*2), int(self.patch_len*self.scale*2)))

        high_res_patch = high_res_patch.convert('RGB')
        high_res_patch = high_res_patch.resize((512, 512))
        high_res_patch = transforms.ToTensor()(high_res_patch)

        # mpp8 이상을 지원하는 wsi인 경우
        # mpp8 / 현재 mpp, 이 값을 patch(512)에 곱해주면 현재 mpp에서 추출해야 할 patch의 길이를 구할 수 있습니다.
        if self.slide_level_index[1] != 0:
            low_ratio = 8 / self.raw_mpp_x / self.downsamples[1]
            low_res_patch = self.slide.read_region((coord[0] - self.modifier, coord[1] - self.modifier),
                                                   self.slide_level_index[1],
                                                   (int(self.patch_len * low_ratio), int(self.patch_len * low_ratio)))
        # mpp2 이상을 지원하지만, mpp 8이상을 지원하지 않는 wsi일 경우
        # mpp8 / 현재 mpp, 이 값을 patch(512)에 곱해주면 현재 mpp에서 추출해야 할 patch의 길이를 구할 수 있습니다.
        elif self.slide_level_index[0] != 0 and self.slide_level_index[1] == 0:
            low_ratio = 8 / self.raw_mpp_x / self.downsamples[0]
            low_res_patch = self.slide.read_region((coord[0] - self.modifier, coord[1] - self.modifier),
                                                   self.slide_level_index[0],
                                                   (int(self.patch_len * low_ratio),
                                                    int(self.patch_len * low_ratio)))
        # mpp가 2, 8을 둘 다 지원하지 않을 때
        elif self.slide_level_index[0] == 0 and self.slide_level_index[1] == 0:
            low_res_patch = self.slide.read_region((coord[0] - self.modifier, coord[1] - self.modifier),
                                                   self.slide_level_index[1],
                                                   (int(self.patch_len * self.scale * 8),
                                                    int(self.patch_len * self.scale * 8)))

        low_res_patch = low_res_patch.convert('RGB')
        low_res_patch = low_res_patch.resize((512, 512))
        low_res_patch = transforms.ToTensor()(low_res_patch)

        return high_res_patch, low_res_patch, coord[0], coord[1]
