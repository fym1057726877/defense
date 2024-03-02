import torch as th
from ContrastExperiances.Diffusion.models import UNetModel, Memory, ConvEncoder, ConvDecoder


def test_unet():
    device = "cuda"
    unet = UNetModel(
        in_channels=4,
        model_channels=64,
        out_channels=4,
        channel_mult=(1, 2, 4),
        attention_resolutions=(4,),
        num_res_blocks=2,
    ).to(device)
    x = th.randn((8, 4, 16, 16), device=device)
    t = th.randint(0, 100, (8,), device=device)
    out = unet(x, t)
    print(out.shape)


def test_memory():
    x = th.randn((16, 192, 7, 7))
    model = Memory(MEM_DIM=200, features=192 * 7 * 7, addressing="sparse")
    out = model(x)
    print(out[0].shape, out[1].shape)


def testencoder():
    e = ConvEncoder(image_channels=1, conv_channels=64)
    x = th.randn((16, 1, 64, 64))
    o = e(x)
    print(o.shape)


def testdecoder():
    d = ConvDecoder(image_channels=1, conv_channels=64)
    x = th.randn((16, 4096))
    o = d(x)
    print(o.shape)


def test():
    output = th.randint(0, 10, (8, 8))
    i = th.randint(0, 8, (4, ))
    b = output[i].view(-1, 2, 4)
    print(b.shape)

if __name__ == '__main__':
    test()