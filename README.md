

### Code Installation

First, clone the repository locally:
```shell
git clone https://github.com/hotonbao/DETR-custom-coco.git
```
Move to the project directory
```shell
cd .\DETR-custom-coco\
```
Dowload COCO dataset format Streeview follow below link:

```shell
https://universe.roboflow.com/fsmvu/street-view-gdogo/dataset/3#
```
Extract the dataset and move to the data folder like below folder tree:
```
root/
└── data/
    ├── test/
    |   ├──_annotations.coco.json 
    |   ├──img1.jpg 
    |   ├──img2.jpg 
    |   ├──...jpg 
    ├── train/
    |   ├──_annotations.coco.json 
    |   ├──img1.jpg 
    |   ├──img2.jpg 
    |   ├──...jpg 
    ├── valid/     
    |   ├──_annotations.coco.json 
    |   ├──img1.jpg 
    |   ├──img2.jpg 
    |   ├──...jpg 
└── detr/
    ├──d2/
    ├──dataset/
    ├──models/
    ├──output/
    ├──ultil
    ├──...

```



Training with data format COCO 6 class vehicles:
- batch size: 8
- epochs: 20
- device: 'cuda' or 'cpu'
- output: save weight directory
- 
```shell
python main.py --dataset_file vehicle --data_path ../data/ --batch_size 8 --epochs 20 --device cuda --output_dir output  
```

