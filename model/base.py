import torch
import torch.nn as nn


# =========================
# SE BLOCK (NEW)
# =========================
class SEBlock(nn.Module):
    def __init__(self, ch, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(ch, ch // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // reduction, ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(self.pool(x))
        return x * w


# =========================
# ORIGINAL MODULES
# =========================
class UIA(nn.Module):
    def __init__(self, channels, ks):
        super().__init__()
        self._c_avg = nn.AdaptiveAvgPool2d((1, 1))
        self._c_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self._c_sig = nn.Sigmoid()

        self._h_avg = nn.AdaptiveAvgPool2d((1, None))
        self._h_conv = nn.Conv2d(channels, channels, 1, groups=channels, bias=False)

        self._w_avg = nn.AdaptiveAvgPool2d((None, 1))
        self._w_conv = nn.Conv2d(channels, channels, 1, groups=channels, bias=False)

        self._hw_conv = nn.Conv2d(
            channels,
            channels,
            ks,
            padding=ks // 2,
            padding_mode="reflect",
            groups=channels,
            bias=False,
        )

        self._chw_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self._chw_sig = nn.Sigmoid()

    def forward(self, x):
        c_weight = self._c_sig(self._c_conv(self._c_avg(x)))
        h_map = self._h_conv(self._h_avg(x))
        w_map = self._w_conv(self._w_avg(x))
        hw_map = self._hw_conv(w_map @ h_map)
        chw_weight = self._chw_sig(self._chw_conv(c_weight * hw_map))
        return chw_weight * x


class NormGate(nn.Module):
    def __init__(self, channels, ks):
        super().__init__()
        self._norm_branch = nn.Sequential(
            nn.InstanceNorm2d(channels),
            nn.Conv2d(
                channels,
                channels,
                ks,
                padding=ks // 2,
                padding_mode="reflect",
                bias=False,
            ),
        )
        self._sig_branch = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                ks,
                padding=ks // 2,
                padding_mode="reflect",
                bias=False,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self._norm_branch(x) * self._sig_branch(x)


class UCB(nn.Module):
    def __init__(self, channels, ks):
        super().__init__()
        self._body = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                ks,
                padding=ks // 2,
                padding_mode="reflect",
                bias=False,
            ),
            NormGate(channels, ks),
            UIA(channels, ks),
            SEBlock(channels),  # NEW
        )

    def forward(self, x):
        return x + self._body(x)


class PWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self._body = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2,
            padding_mode="reflect",
            bias=False,
        )

    def forward(self, x):
        return self._body(x)


# =========================
# IMPROVED COLOR BRANCH
# =========================
class GlobalColorCompensationNet(nn.Module):
    def __init__(self, channel_scale, ks):
        super().__init__()
        self._body = nn.Sequential(
            PWConv(3, channel_scale, ks),
            UCB(channel_scale, ks),
            UCB(channel_scale, ks),
            UCB(channel_scale, ks),
            UCB(channel_scale, ks),  # added
            UCB(channel_scale, ks),  # added
            PWConv(channel_scale, 3, ks),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x + self._body(x)  # residual


# =========================
# FINAL MODEL
# =========================
class CLCC(nn.Module):
    def __init__(self, channel_scale, main_ks, gcc_ks):
        super().__init__()

        self._color_branch = GlobalColorCompensationNet(channel_scale, gcc_ks)

        self._in_conv = nn.Sequential(
            PWConv(3, channel_scale, main_ks), UIA(channel_scale, main_ks)
        )

        # deeper groups
        self._group1 = nn.Sequential(*[UCB(channel_scale, main_ks) for _ in range(6)])
        self._group2 = nn.Sequential(*[UCB(channel_scale, main_ks) for _ in range(6)])
        self._group3 = nn.Sequential(*[UCB(channel_scale, main_ks) for _ in range(6)])

        self._group1_adaptation = nn.Sequential(
            PWConv(3, channel_scale, main_ks), UCB(channel_scale, main_ks)
        )
        self._group2_adaptation = nn.Sequential(
            PWConv(3, channel_scale, main_ks), UCB(channel_scale, main_ks)
        )
        self._group3_adaptation = nn.Sequential(
            PWConv(3, channel_scale, main_ks), UCB(channel_scale, main_ks)
        )

        self._out_conv = nn.Sequential(PWConv(channel_scale, 3, main_ks), nn.Tanh())

    def forward(self, x):
        color_comp = 1 - x
        color_map = self._color_branch(color_comp)

        in_feat = self._in_conv(x)

        g1 = self._group1(in_feat)
        g1 = g1 + self._group1_adaptation(color_map * color_comp)

        # cross-stage residuals
        g2_input = g1 + in_feat
        g2 = self._group2(g2_input)
        g2 = g2 + self._group2_adaptation(color_map * color_comp)

        g3_input = g2 + g1
        g3 = self._group3(g3_input)
        g3 = g3 + self._group3_adaptation(color_map * color_comp)

        out = self._out_conv(g3)

        # multi-output
        return [g1, g2, out]
