# 在树莓派4B上运行
# python_3.7.3
# numpy_1.17.0  matplotlib_3.3.4  pillow_8.1.0
# opencv_python_4.1.0.25  opencv_contrib_python_4.1.0.25
# torch_1.3.0  torchvision_0.4.1


import torch
import torch.nn as nn
from torchvision import transforms,models
import time
import os
import cv2 as cv
from PIL import Image
import RPi.GPIO as GPIO
import threading
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif']=['Microsoft YaHei'] #显示中文
from collections import Counter


grabage_name = ['蔬菜','水果','塑料瓶','易拉罐','烟头','陶瓷碎片','电池']
grabage = ['厨余垃圾','可回收垃圾','其他垃圾','有害垃圾']
predict_grabage_class = []

# 拍照图片保存位置
image_path = "/home/pi/Desktop/"

# 模型权重文件位置
model_path = "/home/pi/python/yang_googlenet.pth"


#初始化GPIO输出信号  厨余、可回收、其他、有害（默认低电平，识别到后高电平）
def init_output_gpio(channel_4,channel_5,channel_6,channel_7):
    GPIO.setup(channel_4, GPIO.OUT,initial=GPIO.LOW)
    GPIO.setup(channel_5, GPIO.OUT,initial=GPIO.LOW)
    GPIO.setup(channel_6, GPIO.OUT,initial=GPIO.LOW)
    GPIO.setup(channel_7, GPIO.OUT,initial=GPIO.LOW)
    
#图片规范化处理
predict_transform = transforms.Compose([
    transforms.Resize((230,230)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])

#加载模型及模型参数
def load_param(model,model_path):
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path,map_location='cpu'),strict=False)
        
#判断垃圾类别，发送给下位机控制舵机
def grabage_class(predicted_num):
    # 蔬菜 水果 --厨余垃圾
    if predicted_num < 2:
        GPIO.output(31,GPIO.HIGH)
        time.sleep(0.15)
        init_output_gpio(31,33,35,37)
        
    # 塑料瓶 易拉罐 --可回收垃圾
    elif predicted_num < 4:
        GPIO.output(33,GPIO.HIGH)
        time.sleep(0.15)
        init_output_gpio(31,33,35,37)
        
    # 烟头 陶瓷碎片  --其它垃圾
    elif predicted_num < 6:
        GPIO.output(35,GPIO.HIGH)
        time.sleep(0.15)
        init_output_gpio(31,33,35,37)

    # 电池  --有害垃圾    
    else:
        GPIO.output(37,GPIO.HIGH)
        time.sleep(0.15)
        init_output_gpio(31,33,35,37)
           
#预测垃圾种类
def img_predict(model,img_input):    
    model.eval()
    with torch.no_grad():
        cam_start = time.time()
        output = model(img_input)
        cam_end = time.time()
        cam_time.append(cam_end-cam_start)
    _,predicted = torch.max(output.data,1)
    predicted_num = predicted.item()
    #print(grabage_name[predicted_num])  #垃圾名字
    
    # 打印每次的信息：  1 可回收垃圾 1 OK！
    global i
    if predicted_num < 2:
        print_class = "厨余垃圾"
        predict_grabage_class.append(print_class)
    elif predicted_num < 4:
        print_class = "可回收垃圾"
        predict_grabage_class.append(print_class)
    elif predicted_num < 6:
        print_class = "其它垃圾"
        predict_grabage_class.append(print_class)
    else:
        print_class = "有害垃圾"
        predict_grabage_class.append(print_class)

    count = Counter(predict_grabage_class)
    print("%d  %s  %d  OK!"%(i+1,print_class,count[print_class]))
    #print("\r            ",end="",flush = True)
    
    grabage_class(predicted_num)    


# 可视化显示结果信息
def output_show():
    fig = plt.figure(figsize=(20,10))
    fig.canvas.set_window_title('Output')

    # 1-12为垃圾图片 
    for i in range(12):
        plt.subplot(3,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("%s：%s"%(str(i+1),predict_grabage_class[i]),fontsize=20)
        img = Image.open(image_path+"%d.jpg"%i)
        img = img.resize((180,110))
        plt.imshow(img)
    
    # 13为圆饼图
    count = Counter(predict_grabage_class)
    x = [count["厨余垃圾"],count["可回收垃圾"],count["其他垃圾"],count["有害垃圾"]]
    explode = (0.05,0,0,0)
    plt.subplot(3,5,13)
    plt.xticks([])
    plt.yticks([])    
    plt.pie(x,labels=grabage,radius=0.8,autopct="%.0f%%",textprops={"fontsize":12,"color":"k"},explode=explode,shadow=True,startangle=20,pctdistance=0.5)
    plt.xlabel("圆饼图",fontsize=20)
    plt.axis("equal")
   
    # 14为信息
    plt.subplot(3,5,14)
    plt.xticks([])
    plt.yticks([])
    #plt.xlabel("xxxxx",fontsize=20)  
    plt.text(0.065,0.300,"总用时：%.2f秒"%(end_time-start_time),fontsize=20)

    plt.show()


if __name__ == "__main__":
    yang = models.googlenet(pretrained = False)
    yang.fc = nn.Linear(in_features=1024,out_features=7,bias=True)
    
    GPIO.setmode(GPIO.BOARD)
    GPIO.setwarnings(False)
    GPIO.setup(11, GPIO.IN)            # 红外传感器  拍照
    init_output_gpio(31,33,35,37)      # 初始化信号输出引脚
    
    i = 0

    cam_time = []#0.3秒
    load_param(yang,model_path=model_path)
    print("开始检测")
    while(1):    
        if GPIO.input(11) == 0:
            if i == 0:
                start_time = time.time()

            cap = cv.VideoCapture(0)            
            ret,frame = cap.read()
            cap.release()
            cv.imwrite(image_path+"%d.jpg"%i,frame)
            #print(image_path+"%d.jpg"%i)
            img = Image.open(image_path+"%d.jpg"%i)
            img_tensor = predict_transform(img)
            img_input = torch.unsqueeze(img_tensor,0)
            img_predict(yang,img_input)
            
            i += 1

            if i == 12:
                end_time = time.time()
                break
            
    output_show()
    GPIO.cleanup()
    cap.release()               
    #print(cam_time)
    while(1):
        pass
