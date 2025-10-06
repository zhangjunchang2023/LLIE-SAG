Low-light  Image Enhancement Integrated Semantic Aware Guidance


1. Create Environment
   
Install the environment with Pytorch 1.10

Make Conda Environment

conda create -n LOIE python=3.8

conda activate LOIE

Install Dependencies

conda install pytorch=1.11 torchvision cudatoolkit=11.3 -c pytorch

2. Prepare Dataset
   
 Synthetic datasets from CamVid and Cityscapes：
 
  https://pan.baidu.com/s/1j7-iDqp2UccTNhZ10eES6A?pwd=nw5w password: nw5w
  
Datasets without reference images -NPE, LIME, MEF, DICM and VV:

  https://pan.baidu.com/s/1zlMuewN4LEViqEVWK0dNdQ?pwd=jn99 password:jn99
  ![image](https://github.com/user-attachments/assets/4fac8b5b-3470-45e0-83bf-bfb1ccf44933)

  
Figure 7 corresponds to a dataset of real 106 images taken randomly at night:

 https://pan.baidu.com/s/1yg9zgktfQx2RDIM-tUr5iw?pwd=sdzi password:sdzi

3. Test

To begin:

download  https://pan.baidu.com/s/1sAuyqkIJw_zjy_7B28VJrA?pwd=ia46 password: ia46, put it in the segmodel_checkpoint folder.

download  https://pan.baidu.com/s/1FPugqknmLsS0A0Ga_J1ehw?pwd=qq5g password: qq5g put it in the checkpoint folder.

Runs: LOIE_utils/Evaluate_images.py
