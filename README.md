# Road_characteristic_estimation_using_deep_learning

## Proposed method
We propse the method to estimate the road characteristic using the intermediate layer of deep learning for autonomous car.<br>
In the research, we estimate the road suface vibration characteristics from the image by mapping the foward road surface image to the vibration transmitted to the vehicle on the road.<br>
Specifically, we use the deep learning model using road surface as the input and the vibration as the output and estimates the road surface characteristics by the intermediate layer of the model when the road suface image is input.<br>
<image src="pictures/encoder_decoder.png" width=50%>
In the process of generating vibration from high-dimensional image, we think that the only main vibration information can be obtained from the image by passing through the low-dimensional intermediate layer.

## Target for verification
e perform the experiment on a small vehicle to verify whether the low-dimensional intermediate layer on the road suface show 3 points below.
<image src="pictures/road.png" width=50%>

1. Each road's latent variables of intermediate layer are clusterd or not
2. Each road's latent variables of intermediate layer suggests the possibility of continuous expression or not
3. Each road's latent variables of intermediate layer expresses the vibration information or not (e.g., wheter the clusters of the road surface with small vibration close or not)

## Parameters of the model used in the verification
<image src="pictures/encoder_decoder_exact.png" width=50%>
  
## Results
You make the model from the image and vibration and preserve it
``` 
  bash YPKerberos.sh
```
You output the latent variables when the test image are input to the saved model
``` 
  bash YPKerberos.sh
```
<image src="pictures/latent_space.png" width=50%>


## 提案手法
前方路面画像とその路面での車体に伝わる振動を対応付けることで，画像から路面振動特性を推定する手法を提案する．<br>
具体的には，路面画像を入力，振動を出力として深層学習モデルを生成し，路面画像が入力された際のモデルの中間層により路面特性を推定する手法を提案する．<br>
高次元の画像情報から振動情報を生成する過程において低次元の中間層を通すことにより，画像から振動の主要な情報のみを取得できるのではないかと考える．

## 検証内容
小型移動体において実験を行い，の路面において低次元の中間層が検証する．<br>
今回，の路面で低次元の中間層が次のようになっているか検証する． <br>
・路面ごとに散らばっているか <br>
・連続的に表現できる可能性を示唆しているか <br>
・振動情報が表現されているか（例えば，振動が小さい路面同士が近い場所に存在しているか） 

## 本検証で用いたパラメータ


## 検証結果
画像情報，振動情報を取得して，学習を回し，モデルを保存する．
``` 
  bash YPKerberos.sh
```
保存したモデルに対し，学習に用いていないテスト画像を入力し，潜在変数を出力する．
``` 
  bash YPKerberos.sh
```

