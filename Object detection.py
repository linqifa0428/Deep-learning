import torch
from d2l import torch as d2l


d2l.set_figsize()
img=d2l.plt.imread('C:/Users/28227/Desktop/2.jpg')
d2l.plt.imshow(img)

def box_corner_to_center(boxes):
    """从（左上，右下）转换到（中间，宽度，高度）"""
    x1,y1,x2,y2=boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    cx=(x1+x2)/2
    cy=(y1+y2)/2
    w=x2-x1
    h=y2-y1
    boxes=torch.stack((cx,cy,w,h),axis=-1)
    return boxes

def box_center_to_corner(boxes):
    """从（中间，宽度，高度）转换到（左上，右下）"""
    cx,cy,w,h=boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    x1=cx-w/2
    y1=cy-h/2
    x2=cx+w/2
    y2=cy+h/2
    boxes=torch.stack((x1,y1,x2,y2),axis=-1)
    return boxes

dog_bbox,cat_bbox=[60,45,378,516],[400,112,655,493]

boxes=torch.tensor((dog_bbox,cat_bbox))
print(box_center_to_corner(box_corner_to_center(boxes))==boxes)

def bbox_to_rect(bbox,color):
    return d2l.plt.Rectangle(
        xy=(bbox[0],bbox[1]),width=bbox[2]-bbox[0],height=bbox[3]-bbox[1],
        fill=False,edgecolor=color,linewidth=2)

fig=d2l.plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox,'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox,'red'))
d2l.plt.show()  # 添加这行代码以显示图像