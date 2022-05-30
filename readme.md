# Before running the code ...
downlad the data from https://drive.google.com/file/d/1YZQFSonulXuagMIfbJkZeTFJ6qEUuUxL/view?usp=sharing and extract files to a ./data folder

Do something like this
.  
├── data  
│   ├── dataset.json  
│   ├── imagesTr  
│   ├── imagesTs  
│   └── labelsTr  
├── V_NAS.py  
├── main.py  
...  
...  
...  

# V_NAS.py

Model articture for VNAS, this model has ability to update specific parameters

# searched_nas.py

Model articture for searched NAS, you can try differnet combination of model(s) by passing the desired block index list to each Encoder/Decoder


# best.py & worst.py

Train the best or worst model

# search.py 

Differentialable articture search