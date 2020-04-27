# Road_characteristic_estimation_using_deep_learning

## Proposed method
We propse the method to estimate the road characteristic using the intermediate layer of deep learning for autonomous car.<br>
In the research, we estimate the road suface vibration characteristics from the image by mapping the foward road surface image to the vibration transmitted to the vehicle on the road.<br>

Specifically, we use the deep learning model using road surface as the input and the vibration as the output and estimates the road surface characteristics by the intermediate layer of the model when the road suface image is input.<br>

In the process of generating vibration from high-dimensional image, we think that the only main vibration information can be obtained from the image by passing through the low-dimensional intermediate layer.
<image src="pictures/encoder_decoder.png" width=80%>

## Target for verification
We perform the experiment on a small vehicle to verify whether the low-dimensional intermediate layer on the road suface of the picture below satisfies 3 points.

<image src="pictures/Road.png" width=80%>

1. Each road's latent variables of intermediate layer are clusterd or not
2. Each road's latent variables of intermediate layer suggests the possibility of continuous expression or not
3. Each road's latent variables of intermediate layer expresses the vibration information or not (e.g., the clusters of the road surface with small vibration close or not)

## Parameters of the model used in the verification
<image src="pictures/encoder_decoder_exact.png" width=80%>
  
## Results
First, you make the model from the image (X_train) and vibration (y_train) and preserve it.
In this envrironment, tensorflow version is 2.1.0 and python version is 3.5.2.
``` 
  python model.py
```
Then, you output the latent variables of the picture below when the test images are input to the saved model.

<image src="pictures/latent_space.png" width=40%>


