import time
import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from classifier.attackClassifier import advAttack
from torch.utils.data import DataLoader, TensorDataset
from classifier.attackClassifier import generateAdvImage


def get_project_path(project_name):
    """
    :param project_name: 项目名称，如pythonProject
    :return: ******/project_name
    """
    # 获取当前所在文件的路径
    cur_path = os.path.abspath(os.path.dirname(__file__))

    # 获取根目录
    return cur_path[:cur_path.find(project_name)] + project_name


def draw_img_groups(img_groups: list, imgs_every_row: int = 6, block: bool = True, show_time: int = 5):
    num_groups = len(img_groups)
    channel = img_groups[0].shape[1]
    for i in range(num_groups):
        assert img_groups[i].shape[0] >= imgs_every_row
        if channel == 1:
            img_groups[i] = img_groups[i].cpu().squeeze(1).detach().numpy()
        else:
            img_groups[i] = img_groups[i].permute(0, 2, 3, 1).cpu().detach().numpy()
    fig = plt.figure()
    gs = fig.add_gridspec(num_groups, imgs_every_row)
    for i in range(num_groups):
        for j in range(imgs_every_row):
            ax = fig.add_subplot(gs[i, j])
            if channel == 1:
                ax.imshow(img_groups[i][j], cmap="gray")
            else:
                ax.imshow(img_groups[i][j])
            ax.axis("off")
    plt.tight_layout()
    plt.show(block=block)
    if not block:
        plt.pause(show_time)
        plt.close("all")


def print_time_consumed(seconds, get=False):
    minutes = seconds // 60
    remain_seconds = seconds % 60
    hours = minutes // 60
    remain_minutes = minutes % 60
    info = f"time consumed: {hours}h{remain_minutes}min{remain_seconds}s"
    print(info)
    if get:
        return info


def show_adv_img(img, target_classifier, attack_type, eps):
    target_classifier.eval()
    if isinstance(eps, float):
        adv_img = advAttack(target_classifier, img, attack_type, eps)
        draw_img_groups([img, adv_img])
    elif isinstance(eps, (list, tuple)):
        img_groups = [img]
        for e in eps:
            adv_img = advAttack(target_classifier, img, attack_type, e)
            img_groups.append(adv_img)
        draw_img_groups(img_groups)


def drow_plot_curve(data: list):
    data = np.array(data)
    x = range(len(data))
    # 创建损失曲线图
    plt.plot(x, data, label='Loss')
    # 添加标签和标题
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    # 添加图例
    plt.legend()
    # 显示图形
    plt.show()


def rot90(x):
    # x: b,1,h,w
    # 获取图像的高度和宽度
    x = x.squeeze(1)
    height, width = x.shape[1:]

    # 创建一个新的张量来存储旋转后的图像
    rotated_tensor_image = torch.zeros_like(x, device=x.device)

    # 逐像素进行旋转
    for i in range(height):
        for j in range(width):
            rotated_tensor_image[:, j, height - 1 - i] = x[:, i, j]

    return rotated_tensor_image.unsqueeze(1)


def defense(x, defense_model):
    rec_x = None
    out = defense_model(x)
    if isinstance(out, tuple):
        rec_x = out[0]
    if isinstance(out, torch.Tensor):
        rec_x = out
    return rec_x


def accuracy(y, y1):
    return (y.max(dim=1)[1] == y1).sum().data.item()


def simple_test_defense(*, defense_model, classifier=None, classifier_name="ModelB_PV600", eps=0.1,
                        defense_fn=defense, attack_type="FGSM", device="cuda", progress=True, testloader=None,
                        print_info=True, iters=30, logits=False):
    # assert attack_type in ["FGSM", "PGD"]
    if classifier is None:
        try:
            model_function_name = f"load_{classifier_name}"
            module = __import__("classifier.models", fromlist=[model_function_name])
            # 从模块中获取函数对象
            model_function = getattr(module, model_function_name)
            classifier = model_function(device=device, print_info=print_info)
        except ImportError:
            print("cannot import model_function_name from classifier.models")

    assert classifier is not None
    # adversial data
    adv_folder_path = os.path.join(get_project_path("Defense"), "data", f"adv_imgs_{classifier_name}")
    if not os.path.exists(adv_folder_path):
        os.makedirs(adv_folder_path)

    adv_imgs_path = os.path.join(get_project_path("Defense"), "data", f"adv_imgs_{classifier_name}",
                                 f"{classifier_name}_{attack_type}_{eps}.pth")

    def getAdvDataLoader(shuffle=False, batch_size=50):
        if os.path.exists(adv_imgs_path):
            data_dict = torch.load(adv_imgs_path)
        else:
            assert testloader is not None
            data_dict = generateAdvImage(
                classifier=classifier,
                attack_dataloader=testloader,
                attack_type=attack_type,
                eps=eps,
                iters=iters,
                progress=True,
                savepath=adv_imgs_path,
                logits=logits
            )
        normal_data = data_dict["normal"]
        adv_data = data_dict["adv"]
        label = data_dict["label"]
        dataloder = DataLoader(TensorDataset(normal_data, adv_data, label), batch_size=batch_size, shuffle=shuffle)
        return dataloder

    advDataloader = getAdvDataLoader(shuffle=True)

    # with torch.no_grad():
    normal_acc, rec_acc, adv_acc, rec_adv_acc, num = 0, 0, 0, 0, 0
    total_num = len(advDataloader)

    iterObject = enumerate(advDataloader)
    if progress:
        iterObject = tqdm(iterObject, total=total_num, desc="testing")

    for i, (img, adv_img, label) in iterObject:
        img, label = img.to(device), label.to(device)
        num += label.size(0)

        normal_y = classifier(img)
        # print(normal_y[0:5])
        normal_acc += accuracy(normal_y, label)
        # print(f"normal_acc:{normal_acc}/{num}")

        rec_img = defense_fn(img, defense_model)
        rec_y = classifier(rec_img)
        rec_acc += accuracy(rec_y, label)
        # print(f"rec_acc:{rec_acc}/{num}")

        adv_img = adv_img.to(device)
        adv_y = classifier(adv_img)
        # print(adv_y[0:5])
        adv_acc += accuracy(adv_y, label)
        # return
        # print(f"adv_acc:{adv_acc}/{num}")

        rec_adv_img = defense_fn(adv_img, defense_model)
        rec_adv_y = classifier(rec_adv_img)
        rec_adv_acc += accuracy(rec_adv_y, label)
        # print(f"rec_adv_acc:{rec_adv_acc}/{num}")

    NorAcc = round(normal_acc / num, 3)
    RecAcc = round(rec_acc / num, 3)
    AdvAcc = round(adv_acc / num, 3)
    RAvAcc = round(rec_adv_acc / num, 3)
    print("test adv_imgs_path: ", adv_imgs_path)
    log = {"attacktype": attack_type,
           "eps": eps,
           "NorAcc": NorAcc,
           "RecAcc": RecAcc,
           "AdvAcc": AdvAcc,
           "RAvAcc": RAvAcc,
           "classifier": classifier_name}
    return log


def get_time_cost(*, defense_model, defense_fn=defense, classifier_name="ModelB_PV600",
                  attack_type="FGSM", eps=0.1, device="cuda", progress=True):
    # adversial data
    adv_folder_path = os.path.join(get_project_path("Defense"), "data", f"adv_imgs_{classifier_name}")
    if not os.path.exists(adv_folder_path):
        os.makedirs(adv_folder_path)

    adv_imgs_path = os.path.join(get_project_path("Defense"), "data", f"adv_imgs_{classifier_name}",
                                 f"{classifier_name}_{attack_type}_{eps}.pth")
    if os.path.exists(adv_imgs_path):
        data_dict = torch.load(adv_imgs_path)
    else:
        print(f"no such file:{adv_imgs_path}")
        return
    adv_data = data_dict["adv"]
    label = data_dict["label"]
    total = adv_data.shape[0]
    print(total)
    advDataloader = DataLoader(TensorDataset(adv_data, label), batch_size=50, shuffle=False)

    start_time = time.time()
    total_num = len(advDataloader)
    iterObject = enumerate(advDataloader)
    if progress:
        iterObject = tqdm(iterObject, total=total_num, desc="testing")

    for i, (adv, lab) in iterObject:
        adv, lab = adv.to(device), lab.to(device)
        rec_adv_img = defense_fn(adv, defense_model)

    end_time = time.time()
    seconds = end_time - start_time
    ms = seconds / total * 1000
    return ms
