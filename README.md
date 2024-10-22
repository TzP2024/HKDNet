# HKDNet
  The aim of RGB-D Co-salient object detection (RGB-D Co-SOD) is to locate the most prominent objects within a provided collection of correlated RGB and depth images. The development of the Transformer has resulted in significant advancements in RGB-D Co-SOD. However, existing methods overlook the considerable computational and parametric costs associated with using the Transformer. Although compact models are computationally efficient, they suffer from performance degradation, which limits their practical applicability. This is because the reduction of model parameters weakens their feature representation capability. To bridge the performance gap between compact and complex models, we propose a hybrid knowledge distillation (KD) network, HKDNet-S*, to perform the RGB-D Co-SOD task. This method incorporates positive-negative logits approximation KD to guide the student network (HKDNet-S) in effectively learning the interrelationships among samples with multiple attributes by considering both positive and negative logits. HKDNet-S* primarily consists of the group cosaliency semantic exploration module and the positive and negative logits approximation KD method. Specifically, we employ a trained RGB-D Co-SOD model as a teacher model (HKDNet-T) to train the HKDNet-S with a limited number of participants using KD. Through extensive experiments on three challenging benchmark datasets (RGBD CoSal1k, RGBD CoSal150, and RGBD CoSeg183), we demonstrate that HKDNet-S* achieves superior accuracy while utilizing fewer parameters in comparison to the existing state-of-the-art methods.
# Requirements
Python 3.7+, Pytorch 1.5.0+, Cuda 10.2+, TensorboardX 2.1, opencv-python If anything goes wrong with the environment, please check requirements.txt for details.
# Architecture and Details
![image](https://github.com/TzP2024/HKDNet/blob/main/fig/HKDNet-S.png)

![image](https://github.com/TzP2024/HKDNet/blob/main/fig/HKDNet.png)

# Results
![image](https://github.com/TzP2024/HKDNet/blob/main/fig/table.png)

![image](https://github.com/TzP2024/HKDNet/blob/main/fig/fig.png)
